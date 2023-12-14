import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import timm, os, urllib.request, torch, cv2, weightbridge, imageio
import numpy as np


# Load image
file = os.path.join(os.path.dirname(__file__), "weimaraner.jpg")
if not os.path.isfile(file):
    print("Downloading test image...")
    url = "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02092339_Weimaraner.JPEG?raw=true"
    urllib.request.urlretrieve(url, file)
image = imageio.v3.imread(file)

# Preprocess color
color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
image = image / 255.0
image = (image - color_mean) / color_std

# Crop, resize, permute
s = image.shape[0]
image = image[:s, :s]



configs = [
    ("convnextv2_base.fcmae_ft_in22k_in1k", "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt", (224, 224)),
    ("convnext_base.fb_in1k", "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth", (288, 288)),
]

print("                           Expected  class 178")
for name, url, size in configs:
    print(f"Model: {name}")

    # Resize image to model input size
    image2 = image
    image2 = cv2.resize(image2, size)
    image2 = torch.from_numpy(image2).float()
    image2 = image2.permute(2, 0, 1)

    # Apply without pretrained weights
    model = timm.create_model(name, pretrained=False)
    model.eval()
    output = model(image2[np.newaxis])[0]
    print(f"    No pretrained weights: Predicted class {torch.argmax(output, dim=0)}")

    # Apply with timm pretrained weights
    model = timm.create_model(name, pretrained=True)
    model.eval()
    output = model(image2[np.newaxis])[0]
    print(f"    Timm:                  Predicted class {torch.argmax(output, dim=0)}")



    # Create model
    model = timm.create_model(name, pretrained=False)
    model.eval()

    # Download and load original weights
    file = os.path.join(os.path.dirname(__file__), os.path.basename(url))
    if not os.path.isfile(file):
        print("Downloading original weights...")
        urllib.request.urlretrieve(url, file)
    weights = torch.load(file)

    # Load weights through weightbridge
    weights = weightbridge.adapt(weights, model.state_dict())
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})

    # Apply with weightbridge pretrained weights
    output = model(image2[np.newaxis])[0]
    print(f"    Weightbridge:          Predicted class {torch.argmax(output, dim=0)}")