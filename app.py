import streamlit as st
import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path

# --------------------------
# GLOBAL CONFIG
# --------------------------
MODEL_PATH = "best_suture_model_final_poc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (512, 512)
MASK_ALPHA = 0.55


# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# --------------------------
# TRANSFORMS
# --------------------------
def get_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.CLAHE(p=1.0, clip_limit=2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# --------------------------
# OVERLAY MASK
# --------------------------
def overlay_mask_on_image(image_tensor, mask_tensor, alpha=0.6):

    # --------------------
    # 1. De-normalize image
    # --------------------
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    # # --------------------
    # # 2. Apply CLAHE to brighten contrast
    # # --------------------
    # lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    # L, A, B = cv2.split(lab)

    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # L2 = clahe.apply(L)

    # lab_enhanced = cv2.merge([L2, A, B])
    # img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # --------------------
    # 3. Gamma correction → brightens midtones
    # --------------------
    gamma = 0.8   # <--- lower = brighter
    img_gamma = np.power(img_uint8 / 255.0, gamma)
    img_gamma = (img_gamma * 255).astype(np.uint8)

    # --------------------
    # 4. Prepare mask
    # --------------------
    mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255

    img_bgr = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2BGR)

    red_mask = np.zeros_like(img_bgr)
    red_mask[mask == 255] = (0, 0, 255)

    # --------------------
    # 5. Stronger overlay
    # --------------------
    blended = cv2.addWeighted(img_bgr, 0.85, red_mask, 0.75, 0)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return img_uint8, blended, mask



# --------------------------
# SUTURE COUNTING
# --------------------------
def count_sutures_from_mask(mask):
    # Keep only red pixels (mask=255)
    binary = (mask == 255).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(binary)

    # Exclude background (0)
    return num_labels - 1


# --------------------------
# PREDICT SINGLE IMAGE
# --------------------------
def predict_one(image_pil, model):

    orig = np.array(image_pil)
    transform = get_transforms()
    augmented = transform(image=orig)
    image_tensor = augmented["image"].to(DEVICE)

    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0))
        pred_mask = torch.sigmoid(pred[0].squeeze(0))

    original, blended, mask = overlay_mask_on_image(image_tensor, pred_mask, MASK_ALPHA)

    count = count_sutures_from_mask(mask)

    combined = np.concatenate((original, blended), axis=1)

    return combined, count


# --------------------------
# BATCH PREDICTION
# --------------------------
def predict_batch(folder_path, model):

    results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            image_pil = Image.open(img_path).convert("RGB")

            _, count = predict_one(image_pil, model)

            results.append((file, count))

    return results


# --------------------------
# STREAMLIT UI
# --------------------------
st.title("🔍 Suture Segmentation & Counting App")
st.write("Upload an image or folder to segment sutures and count them.")

model = load_model()

option = st.selectbox(
    "Select Mode",
    ["Single Image Prediction", "Batch Folder Prediction"]
)

# --------------------------
# MODE 1: SINGLE IMAGE
# --------------------------
if option == "Single Image Prediction":

    uploaded = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Mask & Count Sutures"):
            combined, count = predict_one(img, model)

            st.image(combined, caption="Original (Left) + Prediction (Right)", use_column_width=True)
            st.success(f"🔴 Sutures Detected: **{count}**")


# --------------------------
# MODE 2: BATCH PREDICTION
# --------------------------
elif option == "Batch Folder Prediction":

    folder = st.text_input("Enter folder path containing images:")

    if folder and os.path.isdir(folder):
        if st.button("Run Batch Prediction"):
            results = predict_batch(folder, model)

            st.write("### Results:")
            for fname, cnt in results:
                st.write(f"📌 **{fname}** — Sutures: **{cnt}**")

            st.success("Batch processing complete!")


st.markdown("---")
st.write("Developed by Shivam & Satyam 🧵👁️")
