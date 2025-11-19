# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from preprocess_inference import load_image_to_tensor
from models_inference import load_model
from PIL import Image
import io

st.set_page_config(page_title="MRI Tumor Classifier", layout="centered")
st.title("MRI Tumor Type Predictor (4 classes)")
st.write("Upload an MRI scan image. The model predicts one of 4 tumor classes.")

WEIGHTS_PATH = "best_model.pth"

@st.cache_resource
def get_model(weights_path):
    try:
        model, classes, device = load_model(weights_path)
        return model, classes, device, None
    except Exception as e:
        return None, None, None, str(e)

model, classes, device, load_error = get_model(WEIGHTS_PATH)

if load_error:
    st.error(f"Error loading model: {load_error}")
    st.info("Train the model first with train.py to create 'best_model.pth' in this folder.")
    st.stop()
else:
    st.success(f"Model loaded. Classes: {classes}")

uploaded_file = st.file_uploader("Upload MRI image (png/jpg).", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    # show uploaded image
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    with st.spinner("Running inference..."):
        img_tensor = load_image_to_tensor(uploaded_file).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy().ravel()
            # get top predictions
            topk = min(4, len(probs))
            ranked = sorted([(i, float(p)) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)[:topk]

    st.markdown("### Predictions")
    for idx, score in ranked:
        cls_name = classes[idx] if classes is not None and idx < len(classes) else str(idx)
        st.write(f"- **{cls_name}** : {score*100:.2f}%")

    best_idx = ranked[0][0]
    st.success(f"Predicted: **{classes[best_idx]}** ({ranked[0][1]*100:.2f}%)")
