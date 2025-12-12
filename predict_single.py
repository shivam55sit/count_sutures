import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.image as mpimg

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"best_suture_model_final_poc.pth"

# <<< PUT YOUR INPUT IMAGE PATH HERE >>>
IMAGE_PATH = r"test_images\ADL-P55866_Nagose Chandu__30_3_22 9_23 AM_003.JPG"

# <<< OUTPUT FILE NAME >>>
OUTPUT_PATH = "prediction_output.png"

CONFIG = {
    "IMG_SIZE": (512, 512),
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "ENCODER_NAME": "timm-efficientnet-b0",
    "MASK_ALPHA": 0.55,
}


# -----------------------------
# TRANSFORMS
# -----------------------------
def get_transforms():
    return A.Compose([
        A.Resize(CONFIG['IMG_SIZE'][0], CONFIG['IMG_SIZE'][1]),
        A.CLAHE(p=1.0, clip_limit=2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# -----------------------------
# OVERLAY MASK FUNCTION
# -----------------------------
def overlay_mask_on_image(image_tensor, mask_tensor, alpha=0.6):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert C,H,W -> H,W,C then denormalize
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    # Mask to binary
    mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Convert to BGR
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Create red overlay mask
    red_mask = np.zeros_like(img_bgr)
    red_mask[mask == 255] = (0, 0, 255)

    # Alpha blend
    blended = cv2.addWeighted(img_bgr, 1 - alpha, red_mask, alpha, 0)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return img_uint8, blended


# -----------------------------
# PREDICT ONE IMAGE
# -----------------------------
def predict_single_image():

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ ERROR: IMAGE NOT FOUND: {IMAGE_PATH}")
        return

    print("\nLoading model...")
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG['ENCODER_NAME'],
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=CONFIG["DEVICE"]))
    model.to(CONFIG["DEVICE"])
    model.eval()

    print("Reading image...")
    orig = cv2.imread(IMAGE_PATH)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    transform = get_transforms()
    augmented = transform(image=orig)
    image_tensor = augmented["image"].to(CONFIG["DEVICE"])

    print("Performing prediction...")
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0))
        pred_mask = torch.sigmoid(pred[0].squeeze(0))

    print("Generating overlay...")
    original, blended = overlay_mask_on_image(
        image_tensor, pred_mask, alpha=CONFIG["MASK_ALPHA"]
    )

    # Side-by-side comparison
    combined = np.concatenate((original, blended), axis=1)

    print(f"Saving to {OUTPUT_PATH}...")
    mpimg.imsave(OUTPUT_PATH, combined)

    print("\n✅ DONE! Final output image saved:")
    print(OUTPUT_PATH)


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    predict_single_image()
