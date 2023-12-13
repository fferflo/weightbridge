import numpy as np
import hashlib, pickle

def load_pytorch(file: str):
    """Loads weights from a file saved using PyTorch.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """

    import torch
    pth = torch.load(file, map_location=torch.device("cpu"))
    if "state_dict" in pth:
        pth = pth["state_dict"]
    elif "model_state" in pth:
        pth = pth["model_state"]
    elif "model" in pth:
        pth = pth["model"]

    pth = {k: np.asarray(v) for k, v in pth.items() if not k.endswith("num_batches_tracked")}

    return pth

def load_tensorflow(file):
    """Loads weights from a file saved using Tensorflow.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """

    import tensorflow as tf
    ckpt = tf.train.load_checkpoint(file)
    ckpt_names = list(ckpt.get_variable_to_shape_map().keys())
    ckpt = {n: np.asarray(ckpt.get_tensor(n)) for n in ckpt_names}

    return ckpt

def load_numpy(file):
    """Loads weights from a file saved using Numpy.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """
    npz = np.load(file)
    npz = dict(npz)
    return npz

def load_jittor(file):
    """Loads weights from a file saved using Jittor.

    Args:
        file: File to load weights from.

    Returns:
        A dictionary ``{name: weight}`` with loaded weights.
    """
    if file.endswith(".pkl"):
        # See: https://github.com/Jittor/jittor/blob/607d13079f27dd3a701a2c6b7a0236c66bba0699/python/jittor/__init__.py#L81
        with open(file, "rb") as f:
            s = f.read()
        if s.endswith(b"HCAJSLHD"):
            checksum = s[-28:-8]
            s = s[:-28]
            if hashlib.sha1(s).digest() != checksum:
                raise ValueError(f"Pickle checksum does not match! path: {file}\nThis file maybe corrupted, please consider remove it and re-download.")
        try:
            pkl = pickle.loads(s)
        except Exception as e:
            msg = str(e)
            msg += f"\nPath: \"{file}\""
            if "trunc" in msg:
                msg += "\nThis file maybe corrupted, please consider remove it and re-download."
            raise RuntimeError(msg)
    else:
        raise ValueError(f"Invalid file format {file}")

    pkl = {k: np.asarray(v) for k, v in pkl.items() if not k.endswith("num_batches_tracked")}

    return pkl