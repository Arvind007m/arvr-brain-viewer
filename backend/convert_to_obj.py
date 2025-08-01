import nibabel as nib
import numpy as np
from skimage import measure
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import trimesh
import os

def smooth_tumor_mask(tumor_data, sigma=1.5):
    """
    Apply Gaussian smoothing to tumor mask to create smooth surfaces.
    
    Args:
        tumor_data: Binary tumor mask data
        sigma: Gaussian smoothing parameter (higher = smoother)
    
    Returns:
        Smoothed tumor data as float values
    """
    # Convert binary mask to float for smoothing
    tumor_float = tumor_data.astype(np.float32)
    
    # Apply Gaussian smoothing
    smoothed_tumor = gaussian_filter(tumor_float, sigma=sigma)
    
    # Normalize to 0-1 range
    if smoothed_tumor.max() > 0:
        smoothed_tumor = smoothed_tumor / smoothed_tumor.max()
    
    return smoothed_tumor

def smooth_mesh_laplacian(vertices, faces, iterations=5, lambda_factor=0.5):
    """
    Apply Laplacian smoothing to mesh vertices.
    
    Args:
        vertices: Mesh vertices array
        faces: Mesh faces array  
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing strength (0-1, higher = more smoothing)
    
    Returns:
        Smoothed vertices array
    """
    vertices = vertices.copy()
    
    for _ in range(iterations):
        # Create adjacency information
        vertex_neighbors = [[] for _ in range(len(vertices))]
        
        # Build neighbor list from faces
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                vertex_neighbors[v1].append(v2)
                vertex_neighbors[v2].append(v1)
        
        # Apply Laplacian smoothing
        new_vertices = vertices.copy()
        for i, neighbors in enumerate(vertex_neighbors):
            if neighbors:  # If vertex has neighbors
                # Calculate average position of neighbors
                neighbor_positions = vertices[neighbors]
                avg_position = np.mean(neighbor_positions, axis=0)
                
                # Move vertex towards average of neighbors
                new_vertices[i] = vertices[i] + lambda_factor * (avg_position - vertices[i])
        
        vertices = new_vertices
    
    return vertices

def export_obj_with_materials(brain_nii_path, tumor_mask_path, output_obj_path):
    brain_nii = nib.load(brain_nii_path)
    tumor_nii = nib.load(tumor_mask_path)
    brain_data = brain_nii.get_fdata()
    tumor_data = tumor_nii.get_fdata()
    
    verts_all = []
    faces_all = []
    materials = []
    vert_offset = 0

    # Process brain (keep original approach for brain tissue)
    brain_binary = (brain_data > 0).astype(np.uint8)
    if np.sum(brain_binary) > 0:
        print("Processing brain tissue...")
        brain_verts, brain_faces, _, _ = measure.marching_cubes(brain_binary, level=0.5)
        verts_all.extend(brain_verts)
        faces_all.append((brain_faces + vert_offset, 'brain'))
        vert_offset += len(brain_verts)
        materials.append('brain')
    
    # Process tumor with smoothing for better surface quality
    tumor_binary = (tumor_data > 0).astype(np.uint8)
    if np.sum(tumor_binary) > 0:
        print("Processing tumor with smoothing...")
        
        # Apply Gaussian smoothing to tumor mask for smooth surfaces
        smoothed_tumor = smooth_tumor_mask(tumor_binary, sigma=1.5)
        
        # Use marching cubes on smoothed data with appropriate threshold
        tumor_verts, tumor_faces, _, _ = measure.marching_cubes(smoothed_tumor, level=0.3)
        
        # Apply additional mesh smoothing to tumor for ultra-smooth results
        print("Applying mesh smoothing to tumor...")
        smoothed_tumor_verts = smooth_mesh_laplacian(tumor_verts, tumor_faces, iterations=3, lambda_factor=0.4)
        
        verts_all.extend(smoothed_tumor_verts)
        faces_all.append((tumor_faces + vert_offset, 'tumor'))
        vert_offset += len(smoothed_tumor_verts)
        materials.append('tumor')
        print("Tumor smoothing completed.")

    with open(output_obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(output_obj_path).replace('.obj', '.mtl')}\n")
        for v in verts_all:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

        for faces, mat in faces_all:
            f.write(f"usemtl {mat}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    mtl_path = output_obj_path.replace('.obj', '.mtl')
    with open(mtl_path, "w") as f:
        if "brain" in materials:
            f.write("newmtl brain\n")
            f.write("Kd 0.8 0.8 0.8\n")
            f.write("d 0.3\n")
            f.write("illum 2\n")
        if "tumor" in materials:
            f.write("newmtl tumor\n")
            f.write("Kd 1.0 0.0 0.0\n")
            f.write("d 1.0\n")
            f.write("illum 2\n")

    print(f"OBJ and MTL files written to: {output_obj_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    brain_nii = os.path.join(base_dir, "input", "uploaded.nii")
    mask_nii = os.path.join(base_dir, "output", "predicted_mask.nii.gz")
    obj_out = os.path.join(base_dir, "output", "brain_with_tumor.obj")

    export_obj_with_materials(brain_nii, mask_nii, obj_out)