import os
import torch
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from monai.transforms import (
    LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityd, ToTensord, Compose, Resize
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import Dataset

def post_process_segmentation(tumor_mask, brain_data, min_component_size=500, max_tumor_ratio=0.05):
    """
    Post-process the tumor segmentation to remove artifacts and ensure it's within brain boundaries.
    Conservative approach to detect only real tumor core.
    
    Args:
        tumor_mask: Binary tumor segmentation mask
        brain_data: Original brain MRI data  
        min_component_size: Minimum size for connected components to keep
        max_tumor_ratio: Maximum ratio of tumor volume to brain volume (default 5%)
    
    Returns:
        Cleaned tumor mask with realistic tumor sizing
    """
    # Create brain mask (areas with intensity > 0)
    brain_mask = brain_data > 0
    
    # Constrain tumor to be only within brain region
    tumor_mask = tumor_mask & brain_mask
    
    # Remove small isolated components using connected component analysis
    labeled_mask, num_features = ndimage.label(tumor_mask)
    
    if num_features > 0:
        # Get component sizes
        component_sizes = ndimage.sum(tumor_mask, labeled_mask, range(1, num_features + 1))
        
        # For realistic tumor size, keep only the single largest component
        if len(component_sizes) > 0:
            largest_component_idx = np.argmax(component_sizes) + 1
            largest_size = component_sizes[largest_component_idx - 1]
            
            print(f"Found {num_features} components, largest has {largest_size} voxels")
            
            # Only keep the largest component if it's bigger than minimum size
            if largest_size >= min_component_size:
                cleaned_mask = np.zeros_like(tumor_mask)
                cleaned_mask[labeled_mask == largest_component_idx] = True
                tumor_mask = cleaned_mask
                print(f"Keeping single largest tumor component with {largest_size} voxels")
            else:
                print(f"Largest component ({largest_size} voxels) too small, removing all tumor")
                tumor_mask = np.zeros_like(tumor_mask)
        else:
            tumor_mask = np.zeros_like(tumor_mask)
    
    # Check tumor volume relative to brain volume
    brain_volume = np.sum(brain_mask)
    tumor_volume = np.sum(tumor_mask)
    
    if brain_volume > 0:
        tumor_ratio = tumor_volume / brain_volume
        print(f"Tumor to brain volume ratio: {tumor_ratio:.3f}")
        
        # If tumor is too large relative to brain, apply more aggressive filtering
        if tumor_ratio > max_tumor_ratio:
            print(f"Tumor volume too large ({tumor_ratio:.3f} > {max_tumor_ratio}), applying much stricter filtering...")
            
            # Use very aggressive erosion to reduce tumor to core
            structure = ndimage.generate_binary_structure(3, 2)  # More connected structure
            tumor_mask = ndimage.binary_erosion(tumor_mask, structure=structure, iterations=3)
            
            # Re-run connected component analysis - keep only the single largest
            labeled_mask, num_features = ndimage.label(tumor_mask)
            if num_features > 0:
                component_sizes = ndimage.sum(tumor_mask, labeled_mask, range(1, num_features + 1))
                
                # Keep only the single largest component with much higher threshold
                if len(component_sizes) > 0:
                    largest_component_idx = np.argmax(component_sizes) + 1
                    largest_size = component_sizes[largest_component_idx - 1]
                    
                    # Very strict size requirement
                    if largest_size >= min_component_size * 2:
                        cleaned_mask = np.zeros_like(tumor_mask)
                        cleaned_mask[labeled_mask == largest_component_idx] = True
                        tumor_mask = cleaned_mask
                        print(f"After aggressive filtering: keeping {largest_size} voxels")
                    else:
                        print(f"After aggressive filtering: tumor too small ({largest_size} voxels), removing")
                        tumor_mask = np.zeros_like(tumor_mask)
            else:
                tumor_mask = np.zeros_like(tumor_mask)
    
    # Apply enhanced morphological operations for smooth surfaces
    if np.sum(tumor_mask) > 0:  # Only if there's still tumor left
        # Remove small holes (but keep tumor compact)
        tumor_mask = ndimage.binary_fill_holes(tumor_mask)
        
        # Use a spherical structure element for more natural smoothing
        structure_sphere = ndimage.generate_binary_structure(3, 2)  # More connected structure
        
        # Apply opening to remove small protrusions (erosion followed by dilation)
        tumor_mask = ndimage.binary_opening(tumor_mask, structure=structure_sphere, iterations=2)
        
        # Apply closing to fill small gaps and smooth surface (dilation followed by erosion)
        tumor_mask = ndimage.binary_closing(tumor_mask, structure=structure_sphere, iterations=2)
        
        # Apply additional Gaussian smoothing for better surface quality
        # Convert to float, smooth, then threshold back to binary
        tumor_float = tumor_mask.astype(np.float32)
        from scipy.ndimage import gaussian_filter
        smoothed = gaussian_filter(tumor_float, sigma=0.8)
        tumor_mask = (smoothed > 0.5).astype(np.uint8)
        
        # Final gentle erosion to ensure conservative sizing
        structure_small = ndimage.generate_binary_structure(3, 1)  # Less aggressive structure
        tumor_mask = ndimage.binary_erosion(tumor_mask, structure=structure_small, iterations=1)
    
    # Final volume check and report
    final_tumor_volume = np.sum(tumor_mask)
    if brain_volume > 0:
        final_ratio = final_tumor_volume / brain_volume
        print(f"Final tumor to brain volume ratio: {final_ratio:.3f}")
    
    return tumor_mask.astype(np.uint8)

def segment_mri(input_nii_path, output_nii_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_nii = nib.load(input_nii_path)
    original_shape = original_nii.shape
    original_affine = original_nii.affine

    transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ])

    dataset = Dataset(data=[{"image": input_nii_path}], transform=transforms)
    data = dataset[0]
    image = data["image"].unsqueeze(0).to(device)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = sliding_window_inference(image, roi_size=(128, 128, 64), sw_batch_size=1, predictor=model)
        output = torch.sigmoid(output)
        # Use very high threshold for realistic tumor core detection
        output = (output > 0.85).float()

    
    resized_output = Resize(spatial_size=original_shape, mode="nearest")(output[0][0].unsqueeze(0))
    output_np = resized_output.cpu().numpy()[0]
    
    # Load original brain data for post-processing
    brain_data = original_nii.get_fdata()
    
    # Apply post-processing to clean up the segmentation
    print("Applying post-processing to clean up tumor segmentation...")
    print(f"Using very conservative threshold of 0.85 for realistic tumor core detection")
    cleaned_output = post_process_segmentation(output_np > 0, brain_data)

    nib_image = nib.Nifti1Image(cleaned_output, affine=original_affine)
    nib.save(nib_image, output_nii_path)
    print(f"Cleaned segmentation saved and aligned to: {output_nii_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "input", "uploaded.nii")
    model_path = os.path.join(base_dir, "model", "3d_unet_brats2020.pth")
    output_path = os.path.join(base_dir, "output", "predicted_mask.nii.gz")

    segment_mri(input_path, output_path, model_path)

