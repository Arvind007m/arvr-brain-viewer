import os
from segment import segment_mri
from convert_to_obj import export_obj_with_materials
from convert_to_glb import convert_obj_to_glb

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    model_dir = os.path.join(base_dir, "model")

    nii_files = [f for f in os.listdir(input_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
    if not nii_files:
        raise FileNotFoundError("No MRI file found in 'input/' directory.")
    brain_nii_file = nii_files[0]

    brain_path = os.path.join(input_dir, brain_nii_file)
    mask_path = os.path.join(output_dir, "predicted_mask.nii.gz")
    model_path = os.path.join(model_dir, "3d_unet_brats2020.pth")
    obj_path = os.path.join(output_dir, "brain_with_tumor.obj")
    glb_path = os.path.join(output_dir, "brain_with_tumor.glb")

    print("Running segmentation...")
    segment_mri(brain_path, mask_path, model_path)

    print("Generating OBJ and MTL...")
    export_obj_with_materials(brain_path, mask_path, obj_path)

    print("Converting to GLB format...")
    convert_obj_to_glb(obj_path, glb_path)

    print("Pipeline complete!")

if __name__ == "__main__":
    main()