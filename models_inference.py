# models_inference.py
import torch
from model import CNNClassifier_regularization

def load_model(weights_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        classes = checkpoint.get("classes", None)
        num_classes = len(classes) if classes else 4
        model = CNNClassifier_regularization(num_classes=num_classes)
        # load state_dict
        state = checkpoint['state_dict']
        # Handle possible DataParallel 'module.' prefixes
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "")
            new_state[new_key] = v
        model.load_state_dict(new_state)
        model.to(device)
        model.eval()
        return model, classes, device
    else:
        # If someone saved raw state_dict directly
        try:
            state = checkpoint
            # try to infer num_classes from shapes: not safe — default 4
            model = CNNClassifier_regularization(num_classes=4)
            new_state = {}
            for k, v in state.items():
                new_key = k.replace("module.", "")
                new_state[new_key] = v
            model.load_state_dict(new_state)
            model.to(device)
            model.eval()
            return model, None, device
        except Exception as e:
            raise RuntimeError(f"Could not load model: {e}")
