import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- FIX FOR OMP ERROR #15 (Important for Windows) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_PATH = r"best_suture_model_final_poc.pth"
DATA_DIR = r"test_images" # Your data folder
OUTPUT_DIR = "suture_predictions_visuals1" # Folder to save overlay images

CONFIG = {
    "IMG_SIZE": (512, 512),
    "SUTURE_THICKNESS": 6, 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "ENCODER_NAME": "timm-efficientnet-b0", # Must match training config
    "MASK_ALPHA": 0.6 # Transparency level for the overlay
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output images will be saved in: {OUTPUT_DIR}")

# ==========================================
# 2. UPDATED DATASET & TRANSFORMS (Images ONLY)
# ==========================================

class SutureInferenceDataset(Dataset): # Renamed the class for clarity
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Only search for images; ignore JSON files
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")) + 
                                  glob.glob(os.path.join(root_dir, "*.png")))
        
    def __len__(self):
        return len(self.image_paths)

    # Removed _create_mask_from_json since we are only doing inference

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (resize, CLAHE, normalize)
        if self.transform:
            augmented = self.transform(image=image) # Pass image only
            image = augmented['image']
            
        return image, os.path.basename(img_path) # Return tensor and filename

def get_validation_transforms():
    """Only uses the non-destructive transforms (Resize, CLAHE, Normalize)."""
    return A.Compose([
        A.Resize(CONFIG['IMG_SIZE'][0], CONFIG['IMG_SIZE'][1]),
        A.CLAHE(p=1.0, clip_limit=2.0, tile_grid_size=(8, 8)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ==========================================
# 3. VISUALIZATION CORE FUNCTION (No Change)
# ==========================================

def overlay_mask_on_image(image_tensor, mask_tensor, color=(255, 0, 0), alpha=0.6):
    """
    Creates an overlay image from a normalized tensor and a predicted mask.
    """
    # Denormalize image and convert from C, H, W to H, W, C (for OpenCV)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Detach, numpy, transpose C,H,W -> H,W,C, denormalize
    image_np = image_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np * std + mean).clip(0, 1) # Image is now in [0, 1] range
    
    # Convert mask to binary (0 or 1) and then to uint8 (0 or 255)
    mask_np = (mask_tensor.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Prepare for blending (Opencv uses BGR for color ops)
    image_bgr = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    colored_mask = np.zeros_like(image_bgr, dtype=np.uint8)
    # The 'color' tuple is BGR, so we assign it to the regions where mask_np is 255
    colored_mask[mask_np == 255] = color 

    # Alpha Blending (Image, weight, Mask, weight, gamma)
    blended_image_bgr = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)

    # Convert back to RGB for Matplotlib/saving
    return cv2.cvtColor(blended_image_bgr, cv2.COLOR_BGR2RGB)


# ==========================================
# 4. INFERENCE AND SAVING
# ==========================================

def run_inference_and_save(model_path, data_dir, output_dir):
    
    # --- Load Model Structure ---
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG['ENCODER_NAME'], 
        encoder_weights=None, 
        in_channels=3,                  
        classes=1,                      
    )
    
    # --- Load Weights ---
    if not os.path.exists(model_path):
        print(f"Error: Model path not found at {model_path}")
        return
        
    print(f"Loading weights from: {os.path.basename(model_path)}")
    model.load_state_dict(torch.load(model_path, map_location=CONFIG['DEVICE']))
    model.to(CONFIG['DEVICE'])
    model.eval()

    # --- Setup DataLoader ---
    # !!! Using the new Inference Dataset that does not require JSON files !!!
    dataset = SutureInferenceDataset(data_dir, transform=get_validation_transforms())
    
    if len(dataset) == 0:
        print(f"\n🛑 Error: Found 0 images in {data_dir}. Check if files are .jpg or .png.")
        return

    # Use a batch size of 1 for simple visualization
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Starting visualization for {len(dataset)} images...")

    for i, (images, filenames) in enumerate(data_loader):
        image = images[0].to(CONFIG['DEVICE'])
        filename = filenames[0]
        
        with torch.no_grad():
            # Add batch dimension and get prediction
            output = model(image.unsqueeze(0))
            # Apply sigmoid to convert logits to probability mask
            pred_mask = torch.sigmoid(output[0].squeeze(0)) 

        # Create overlay (Suture in Red, BGR format for OpenCV is (0, 0, 255))
        blended_image_rgb = overlay_mask_on_image(
            image.cpu(), 
            pred_mask.cpu(), 
            color=(0, 0, 255), # BGR for red (B=0, G=0, R=255)
            alpha=CONFIG['MASK_ALPHA']
        )
        
        # Save the result
        output_filename = os.path.join(output_dir, f"pred_{filename}")
        mpimg.imsave(output_filename, blended_image_rgb)
        
        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
             print(f"Processed and saved {i+1}/{len(dataset)} images.")

    print(f"\n✅ Visualization complete. Check the '{output_dir}' folder.")
    print("The predicted suture is marked in RED.")

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    run_inference_and_save(MODEL_PATH, DATA_DIR, OUTPUT_DIR)