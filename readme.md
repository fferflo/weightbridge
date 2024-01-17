# weightbridge :bridge_at_night:

### What?

A library to map (deep learning) model weights between different model implementations in Python.

### Why?

Model weights trained using one implementation of an architecture typically cannot directly be loaded into a different implementation of the same architecture, due to:

* Different parameter and layer names.
* Different nesting of modules.
* Different parameter shapes (e.g. `(8, 8)` vs `(64)` vs `(1, 8, 8)`).
* Different order of dimensions (e.g. `(64, 48)` vs `(48, 64)`).
* Different deep learning frameworks (e.g. PyTorch, Tensorflow, Flax).

Adapting the weights manually is a tedious and error-prone process:

```python
k = k.replace('downsample_layers.0.', 'stem.')
k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
k = k.replace('pwconv', 'mlp.fc')
if 'grn' in k:
    k = k.replace('grn.beta', 'mlp.grn.bias')
    v = v.reshape(v.shape[-1])
k = k.replace('head.', 'head.fc.')
if v.ndim == 2 and 'head' not in k:
    model_shape = model.state_dict()[k].shape
    v = v.reshape(model_shape)
```

weightbridge does most of this work for you.

### How?

```python
import weightbridge
new_my_weights = weightbridge.adapt(their_weights, my_weights)
```

* `my_weights` contains the (random) untrained weights created at model initialization (e.g. as the result of `model.state_dict()` in PyTorch, or using `model.init` in Flax and Haiku).
* `their_weights` contains the pretrained weights (e.g. as the result of `torch.load`, `tf.train.load_checkpoint` or `np.load`).

The output has the same structure and weight shapes as `my_weights`, but with the weight values from `their_weights`. It can be used as drop-in for `my_weights`, and for example be stored back into the model using `model.load_state_dict` in PyTorch, or be used in `model.apply` in Flax and Haiku.

**Installation:**

```
pip install weightbridge
```

**Full examples:**

* [examples/mamba2flax.py](https://github.com/fferflo/weightbridge/blob/master/examples/mamba2flax.py): Download weights for [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj) and load into custom Flax implementation.
Print example text.
* [examples/gpt2haiku.py](https://github.com/fferflo/weightbridge/blob/master/examples/gpt2haiku.py): Download weights for [OpenAI GPT-2](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/gpt2) and load into custom Haiku implementation.
Print example text.
* [examples/convnext2timm.py](https://github.com/fferflo/weightbridge/blob/master/examples/convnext2timm.py): Download original weights for [ConvNext](https://github.com/facebookresearch/ConvNeXt) and
[ConvNextV2](https://github.com/facebookresearch/ConvNeXt-V2) and load into
[timm implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py). Test on an example image of ImageNet.
* [examples/vit2flax.py](https://github.com/fferflo/weightbridge/blob/master/examples/vit2flax.py): Download weights for a vision transformer from [torchvision](https://pytorch.org/vision/main/models/vision_transformer.html) and load into
a custom Flax implementation. Test on an example image of ImageNet.

**Additional parameters:**
* `{in_format|out_format}="{pytorch|tensorflow|flax|haiku|...}"` when weights are adapted between different deep learning frameworks (to permute weight axes as required).
* `hints=[...]` to provide additional hints when ambiguous matches cannot be resolved. weightbridge prints an error when this happens, for example:
  ```
  Failed to pair the following nodes
    OUT load_prefix/encode/stage3/block6/reduce/linear/w ((262144,),)
    OUT load_prefix/encode/stage3/block6/expand/linear/w ((262144,),)
    IN  backbone.0.body.layer3.5.conv1.weight ((262144,),)
    IN  backbone.0.body.layer3.5.conv3.weight ((262144,),)
  ```
  We can pass `hints=[("reduce", "conv1")]` (consisting of some uniquely identifying substrings) to resolve the matching failure.
* `cache="some-file"` to store the mapping in a file and reuse it in subsequent calls. If it is not an absolute path, the file is created in the directory of the
  module from which `weightbridge.adapt` is called.
* `verbose=True` to print the matching steps and the final mapping between weights.

weightbridge internally uses a set of heuristics based on the weights' names and shapes to iteratively find mappings between subsets of `my_weights` and `their_weights`, until a unique pairing between all weights is found.

### What does weightbridge not do?

* **Model implementation:** weightbridge does not implement the model, but adapts the weights once the model is implemented (athough it provides a partial sanity-check for the implementation by ensuring that a mapping between the two sets of weights is possible).
When the architecture is implemented using different operations, the weights have to be adapted manually. E.g. in Transformer attention, queries, keys and values can be inferred using different operations:
  ```python
  # Option 1
  x = nn.Linear(features=3 * c)(x)
  q, k, v = jnp.split(x, 3, axis=-1)
  
  # Option 2
  q = nn.Linear(features=c)(x)
  k = nn.Linear(features=c)(x)
  v = nn.Linear(features=c)(x)
  ```
  The corresponding weights have to be split/ concatenated manually and will not be matched by weightbridge otherwise, since it relies on a one-to-one mapping between weights.
* **Hyperparameters:** weightbridge does not ensure that hyperparameters like `nn.LayerNorm(epsilon=1e-6)` or `nn.Conv(padding="SAME")` are set correctly (although some hyperparameters like `use_bias={True|False}` will raise an exception if not set correctly).
