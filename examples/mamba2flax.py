# See: https://github.com/johnma2006/mamba-minimal
import jax.numpy as jnp
import flax.linen as nn
import jax, einx, weightbridge, time, transformers, math, json, torch
import numpy as np
from functools import partial
import einx.nn.flax as einn

# Use channels last layout
Norm = partial(einn.Norm, "... [c]", epsilon=1e-5, mean=False, bias=False) # RMS Norm
Linear = partial(einn.Linear, "... [_|channels]")

class Block(nn.Module):
    kernel_size: int = 4
    d_state: int = 16

    @nn.compact
    def __call__(self, x):
        x0 = x
        x = Norm()(x)

        x = Linear(channels=4 * x.shape[-1], bias=False)(x)
        x, res = jnp.split(x, 2, axis=-1)
        res = jax.nn.silu(res)

        x = nn.Conv(
            features=x.shape[-1],
            kernel_size=(self.kernel_size,),
            feature_group_count=x.shape[-1],
            padding=[(self.kernel_size - 1, self.kernel_size - 1)],
        )(x)[:, :x.shape[1], :]
        x = jax.nn.silu(x) # b l c


        # Compute input-independent state space parameters
        A = self.param("A", nn.initializers.zeros_init(), (x.shape[-1], self.d_state), "float32")
        A = -jnp.exp(A) # c n
        D = self.param("D", nn.initializers.zeros_init(), (x.shape[-1],), "float32") # c

        # Compute input-dependent state space parameters
        dt_rank = math.ceil(x0.shape[-1] / 16)
        y = Linear(channels=dt_rank + self.d_state + self.d_state, bias=False)(x)
        delta, B, C = jnp.split(y, [dt_rank, dt_rank + self.d_state], axis=-1)

        delta = Linear(channels=x.shape[-1], bias=True)(delta)
        delta = jax.nn.softplus(delta) # b l c


        # Scan
        deltaA = jnp.exp(einx.multiply("b l c, c n -> b l c n", delta, A))
        deltaB_x = einx.multiply("b l c, b l n, b l c -> b l c n", delta, B, x)

        def scan(xi, xj):
            deltaA_i, deltaB_x_i = xi
            deltaA_j, deltaB_x_j = xj
            return deltaA_i * deltaA_j, deltaA_j * deltaB_x_i + deltaB_x_j
        _, y = jax.lax.associative_scan(scan, (deltaA, deltaB_x), axis=1)
        y = einx.dot("b l c n, b l n -> b l c", y, C)

        x = y + x * D



        x = x * res
        x = Linear(channels=x0.shape[-1], bias=False)(x)

        x = x0 + x

        return x

class Mamba(nn.Module):
    channels: int
    depth: int
    vocab_size: int = 50280

    @nn.compact
    def __call__(self, x):
        # Vocabulary embedding
        x = einx.get_at("[v] c, b t -> b t c", self.param, x, v=self.vocab_size, c=self.channels)

        # Blocks
        for i in range(self.depth):
            x = Block(name=f"block{i}")(x)
        x = Norm()(x)

        # Classifier
        x = Linear(channels=self.vocab_size, bias=False)(x)

        return x



# Download original weights and config
hf_name = [
    "state-spaces/mamba-130m",
    "state-spaces/mamba-370m",
    "state-spaces/mamba-790m",
    "state-spaces/mamba-1.4b",
    "state-spaces/mamba-2.8b",
    "state-spaces/mamba-2.8b-slimpj",
][-1]

file = transformers.utils.hub.cached_file(hf_name, transformers.utils.CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
with open(file, "r") as f:
    config = json.load(f)

file = transformers.utils.hub.cached_file(hf_name, transformers.utils.WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
pretrained_params = torch.load(file, map_location="cpu")
pretrained_params["backbone.embedding.weight"] = np.transpose(pretrained_params["backbone.embedding.weight"], (1, 0))





text = "We succeeded in taking that picture, and, if you look at it, you see a dot. That's here. That's home. That's us. On it,"
print(f"Input:                 \"{text}\"")

block_size = 512

# Encode text to tokens
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokens = np.asarray(tokenizer(text, return_tensors="np").input_ids)[0]
num_tokens = len(tokens)
tokens = np.pad(tokens, (0, block_size - num_tokens), constant_values=0)

# Create model
rng = jax.random.PRNGKey(int(time.time() * 1000))

model = Mamba(channels=config["d_model"], depth=config["n_layer"])
params = model.init({"params": rng}, tokens[np.newaxis])
apply = jax.jit(model.apply)

def predict(tokens, params, temperature=0.3):
    i = num_tokens
    for _ in range(10): # Predict next tokens
        logits = apply(params, tokens[np.newaxis])[0, i - 1]
        tokens[i] = jax.random.categorical(rng, logits / temperature)
        i += 1
    return tokenizer.decode(tokens[:i])

# Apply without pretrained weights
print(f"No pretrained weights: \"{predict(tokens, params)}\"")

# Map weights to our model implementation
params = weightbridge.adapt(pretrained_params, params, in_format="pytorch", out_format="flax", cache="mamba2flax")

# Apply with pretrained weights
print(f"Weightbridge:          \"{predict(tokens, params)}\"")