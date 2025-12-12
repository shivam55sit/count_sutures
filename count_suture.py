import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import ndimage


# -----------------------------
# TRANSFORMS
# -----------------------------
def get_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.CLAHE(p=1.0, clip_limit=2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# -----------------------------
# BUILD MODEL
# -----------------------------
def load_model(model_path, device):
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -----------------------------
# OVERLAY MASK ON IMAGE
# -----------------------------
def overlay_mask(image_tensor, mask_tensor, alpha=0.55):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255

    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    red_mask = np.zeros_like(img_bgr)
    red_mask[mask == 255] = (0, 0, 255)

    blended = cv2.addWeighted(img_bgr, 1 - alpha, red_mask, alpha, 0)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return img_uint8, blended, mask


# -----------------------------
# IMPROVED SUTURE COUNTING
# -----------------------------
def count_sutures_from_mask(binary_mask, min_length=15, max_gap=30):
    """
    Count sutures by detecting line-like structures in the binary mask.
    
    Args:
        binary_mask: 2D binary mask (255 for suture pixels, 0 for background)
        min_length: Minimum length of a suture to be counted (in pixels)
        max_gap: Maximum gap to bridge when connecting suture fragments
    
    Returns:
        count: Number of detected sutures
        labeled_mask: Visualization of labeled sutures
    """
    
    # Ensure binary
    binary = (binary_mask > 127).astype(np.uint8)
    
    # Apply slight dilation to connect nearby fragments
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
    connected = cv2.dilate(binary, kernel_connect, iterations=1)
    
    # Find connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )
    
    # Filter components based on area and aspect ratio
    valid_sutures = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calculate aspect ratio (length/width for elongated structures)
        max_dim = max(width, height)
        min_dim = min(width, height)
        
        # Filter criteria for suture-like structures:
        # 1. Minimum length
        # 2. Elongated shape (aspect ratio > 2)
        # 3. Reasonable area (not too small noise)
        if max_dim >= min_length and area >= min_length:
            aspect_ratio = max_dim / (min_dim + 1e-6)
            
            # Sutures are typically elongated
            if aspect_ratio > 1.5 or area > 50:
                valid_sutures.append(i)
    
    # Create visualization
    labeled_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    colors = np.random.randint(50, 255, size=(len(valid_sutures),), dtype=np.uint8)
    
    for idx, label_id in enumerate(valid_sutures):
        labeled_mask[labels == label_id] = colors[idx]
    
    return len(valid_sutures), labeled_mask


# Alternative method using skeleton analysis
def count_sutures_skeleton(binary_mask, min_branch_length=20):
    """
    Alternative method: Count sutures by analyzing the skeleton structure.
    This works well for thin, line-like structures.
    """
    from skimage.morphology import skeletonize
    
    # Ensure binary
    binary = (binary_mask > 127).astype(np.uint8)
    
    # Skeletonize to get centerlines
    skeleton = skeletonize(binary)
    skeleton = (skeleton * 255).astype(np.uint8)
    
    # Find endpoints and branch points
    # Convolve with a kernel to find junction points
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton // 255, -1, kernel) * (skeleton // 255)
    
    # Find connected components in skeleton
    num_labels, labels = cv2.connectedComponents(skeleton)
    
    # Count significant branches (filter out noise)
    valid_branches = 0
    for i in range(1, num_labels):
        branch_pixels = np.sum(labels == i)
        if branch_pixels >= min_branch_length:
            valid_branches += 1
    
    return valid_branches


# -----------------------------
# MAIN FUNCTION — CALL THIS
# -----------------------------
def predict_and_count_sutures(image_path, model_path, method='connected_components'):
    """
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        method: 'connected_components' or 'skeleton'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(model_path, device)

    # Read image
    orig = cv2.imread(image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Transform
    transform = get_transforms()
    augmented = transform(image=orig)
    image_tensor = augmented["image"].to(device)

    # Predict mask
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0))
        pred_mask = torch.sigmoid(pred[0].squeeze(0))

    # Make overlay and get binary mask
    original_img, overlay_img, mask_binary = overlay_mask(image_tensor, pred_mask)

    # Count sutures using selected method
    if method == 'skeleton':
        suture_count = count_sutures_skeleton(mask_binary, min_branch_length=15)
        labeled_mask = None
    else:
        suture_count, labeled_mask = count_sutures_from_mask(
            mask_binary, 
            min_length=15,  # Minimum suture length in pixels
            max_gap=8       # Maximum gap to bridge
        )

    print(f"Detected {suture_count} sutures using {method} method")
    
    return overlay_img, suture_count, labeled_mask


# -----------------------------
# USAGE EXAMPLE
# -----------------------------
if __name__ == "__main__":
    image_path = r"test_images\ADL-P55866_Nagose Chandu__30_3_22 9_23 AM_003.JPG"
    model_path = r"best_suture_model_final_poc.pth"
    
    # Try both methods and compare
    print("Method 1: Connected Components")
    overlay1, count1, labeled1 = predict_and_count_sutures(
        image_path, model_path, method='connected_components'
    )
    
    print("\nMethod 2: Skeleton Analysis")
    overlay2, count2, _ = predict_and_count_sutures(
        image_path, model_path, method='skeleton'
    )
    
    print(f"\nFinal counts:")
    print(f"Connected Components: {count1} sutures")
    print(f"Skeleton Analysis: {count2} sutures")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(overlay1)
    axes[0].set_title(f'Overlay (Predicted Mask)')
    axes[0].axis('off')
    
    if labeled1 is not None:
        axes[1].imshow(labeled1, cmap='nipy_spectral')
        axes[1].set_title(f'Labeled Sutures: {count1}')
        axes[1].axis('off')
    
    axes[2].imshow(overlay1)
    axes[2].set_title('Original Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('suture_counting_results.png', dpi=150, bbox_inches='tight')
    plt.show()