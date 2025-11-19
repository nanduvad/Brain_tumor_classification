# preprocess_inference.py
from PIL import Image
from torchvision import transforms
import torch

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image_to_tensor(path_or_file):
    # accepts path string or file-like object (Streamlit upload)
    if hasattr(path_or_file, "read"):
        img = Image.open(path_or_file).convert("RGB")
    else:
        img = Image.open(path_or_file).convert("RGB")

    x = INFER_TRANSFORM(img)
    return x.unsqueeze(0)  # shape (1,3,224,224)
