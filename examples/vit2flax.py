import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import jax.numpy as jnp
from flax import linen as nn
import jax, os, urllib.request, imageio, einx, weightbridge, cv2, math, torchvision
import numpy as np
from functools import partial
import einx.nn.flax as einn

LayerNorm = partial(einn.Norm, "... [c]", epsilon=1e-6)
Linear = partial(einn.Linear, "... [_|channels]")

class Block(nn.Module):
    heads: int = 12
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        # Attention block
        x0 = x
        x = LayerNorm()(x)

        x = Linear(channels=3 * x.shape[-1])(x)
        q, k, v = jnp.split(x, 3, axis=-1)
        q = q * ((q.shape[-1] // self.heads) ** -0.5)

        attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=self.heads)
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

class VisionTransformer(nn.Module):
    channels: int = 768
    patchsize: int = (16, 16)
    depth: int = 12

    @nn.compact
    def __call__(self, x):
        # Patch embedding
        x = nn.Conv(features=self.channels, kernel_size=self.patchsize, strides=self.patchsize, padding=0)(x)

        pos_embed = lambda shape: self.param("pos_embed", nn.initializers.normal(stddev=0.02), shape, "float32")
        x = einx.add("b [s... c]", x, pos_embed)

        class_token = lambda shape: self.param("cls_token", nn.initializers.normal(stddev=0.02), shape, "float32")
        x = einx.rearrange("b s... c, c -> b (1 + (s...)) c", x, class_token)

        # Blocks
        for _ in range(self.depth):
            x = Block()(x)
        x = LayerNorm()(x)

        # Classifier
        x = x[:, 0, :] # Class token
        x = Linear(channels=1000)(x)

        return x





# Load image
file = os.path.join(os.path.dirname(__file__), "weimaraner.jpg")
if not os.path.isfile(file):
    print("Downloading test image...")
    url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02092339_Weimaraner.JPEG?raw=true"
    urllib.request.urlretrieve(url, file)
image = imageio.v3.imread(file)
print("                       Expected  class 178")

# Preprocess color
color_mean = np.asarray([0.5, 0.5, 0.5])
color_std = np.asarray([0.5, 0.5, 0.5])
image = image / 255.0
image = (image - color_mean) / color_std

# Crop, resize
s = image.shape[0]
image = image[:s, :s]
image = cv2.resize(image, (224, 224))


# Create model
model = VisionTransformer()
params = model.init({"params": jax.random.PRNGKey(42)}, jnp.asarray(image[jnp.newaxis]))

# Apply without pretrained weights
output = model.apply(params, image[np.newaxis])[0]
print(f"No pretrained weights: Predicted class {jnp.argmax(output, axis=0)}")


# Download and load original weights
original_params = torchvision.models.vit_b_16(weights="DEFAULT").state_dict()

# The original ViT implementation includes an entry in the positional embedding that is added onto the class token.
# Our implementation appends the class token after positional embedding. We therefore have to add the first entry
# of their positional embedding onto our class token, and remove it from their positional embedding:
pos_embed = original_params["encoder.pos_embedding"]
cls_token = original_params["class_token"]
original_params["class_token"] = cls_token + pos_embed[:, :1, :]
original_params["encoder.pos_embedding"] = pos_embed[:, 1:, :]

# Map weights to our model implementation
params["params"] = weightbridge.adapt(original_params, params["params"], in_format="pytorch", out_format="flax")

# Apply with pretrained weights
output = model.apply(params, image[np.newaxis])[0]
print(f"Weightbridge:          Predicted class {jnp.argmax(output, axis=0)}")