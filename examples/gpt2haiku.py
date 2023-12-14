import jax.numpy as jnp
import haiku as hk
import jax, einx, weightbridge, tiktoken, time, transformers
import numpy as np
from functools import partial
import einx.nn.haiku as einn

LayerNorm = partial(einn.Norm, "... [c]", epsilon=1e-5)
Linear = partial(einn.Linear, "... [_|channels]")

class Block(hk.Module):
    heads: int = 25
    mlp_ratio: int = 4

    def __call__(self, x):
        # Attention block
        x0 = x
        x = LayerNorm()(x)

        x = Linear(channels=3 * x.shape[-1])(x)
        q, k, v = jnp.split(x, 3, axis=-1)
        q = q * ((q.shape[-1] // self.heads) ** -0.5)

        attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads)
        mask = jnp.tril(jnp.ones((q.shape[1], q.shape[1]), dtype=bool)) # Causal mask
        attn = einx.where("q k, b q k h, ", mask, attn, -jnp.inf)
        attn = jax.nn.softmax(attn, axis=-2)
        x = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)

        x = Linear(channels=x.shape[-1])(x)

        x = x + x0

        # MLP block
        x0 = x
        x = LayerNorm()(x)

        x = Linear(channels=x.shape[-1] * self.mlp_ratio)(x)
        x = jax.nn.gelu(x)
        x = Linear(channels=x0.shape[-1])(x)

        x = x + x0

        return x

class GPT2(hk.Module):
    channels: int = 1600
    depth: int = 48
    vocab_size: int = 50257
    block_size: int = 1024

    def __call__(self, x):
        vocab_embed = hk.get_parameter("vocab_embed", (self.vocab_size, self.channels), "float32", hk.initializers.RandomNormal(stddev=0.02))
        x = vocab_embed[x]

        pos_embed = lambda shape: hk.get_parameter("pos_embed", shape, "float32", hk.initializers.RandomNormal(stddev=0.02))
        x = einx.add("b [t c]", x, pos_embed)

        # Blocks
        for i in range(self.depth):
            x = Block(name=f"block{i}")(x)
        x = LayerNorm()(x)

        # Classifier
        x = Linear(channels=self.vocab_size, bias=False)(x)

        return x




text = "We succeeded in taking that picture, and, if you look at it, you see a dot. That's here. That's home. That's us. On it,"
print(f"Input:                 \"{text}\"")

# Encode text to tokens
encoder = tiktoken.get_encoding("gpt2")
tokens = np.asarray(encoder.encode_ordinary(text))
def pad(tokens):
    return np.pad(tokens, (0, GPT2.block_size - len(tokens)), constant_values=0)

# Create model
rng = jax.random.PRNGKey(int(time.time() * 1000))
model = hk.transform(lambda x: GPT2()(x))
params = model.init(rng, pad(tokens)[np.newaxis])
apply = jax.jit(model.apply)

def predict(tokens, params, temperature=0.3):
    i = len(tokens)
    tokens = pad(tokens)
    for _ in range(10): # Predict additional tokens
        logits = apply(params, rng, tokens[np.newaxis])[0, i - 1]
        tokens[i] = jax.random.categorical(rng, logits / temperature)
        i += 1
    return encoder.decode(tokens[:i])

# Apply without pretrained weights
print(f"No pretrained weights: \"{predict(tokens, params)}\"")

# Download original weights
pretrained_params = {k: np.asarray(v) for k, v in transformers.GPT2LMHeadModel.from_pretrained(f"gpt2-xl").state_dict().items()}
pretrained_params["lm_head.weight"] = np.transpose(pretrained_params["lm_head.weight"], (1, 0))
pretrained_params = {k: v for k, v in pretrained_params.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")}

# Map weights to our model implementation
params = weightbridge.adapt(pretrained_params, params, hints=[("norm_1", "ln_2"), ("vocab_embed", "wte")])

# Apply with pretrained weights
print(f"Weightbridge:          \"{predict(tokens, params)}\"")