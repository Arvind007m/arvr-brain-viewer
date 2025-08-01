import os
import trimesh
from trimesh.visual.material import PBRMaterial

def parse_mtl_transparency(mtl_path):
    """Parses .mtl and returns {material_name: alpha}"""
    transparency = {}
    current_mat = None

    if not os.path.exists(mtl_path):
        print(f"MTL file not found: {mtl_path}")
        return {}

    with open(mtl_path, 'r') as f:
        for line in f:
            if line.startswith('newmtl'):
                current_mat = line.strip().split()[1]
            elif line.startswith('d') and current_mat:
                transparency[current_mat] = float(line.strip().split()[1])
            elif line.startswith('Tr') and current_mat and current_mat not in transparency:
                transparency[current_mat] = 1.0 - float(line.strip().split()[1])
    return transparency

def apply_transparency_with_materials(scene, alpha_dict):
    """Applies realistic PBR materials with color and transparency"""
    for name, geom in scene.geometry.items():
        mat_name = geom.visual.material.name if hasattr(geom.visual, 'material') else None
        alpha = alpha_dict.get(mat_name, 1.0)

        if mat_name == "brain":
            mat = PBRMaterial(
                name="brain_material",
                baseColorFactor=[0.8, 0.8, 0.8, alpha],
                alphaMode="BLEND"
            )
        elif mat_name == "tumor":
            mat = PBRMaterial(
                name="tumor_material",
                baseColorFactor=[1.0, 0.0, 0.0, alpha],
                alphaMode="BLEND"
            )
        else:
            mat = PBRMaterial(
                name="default_material",
                baseColorFactor=[1.0, 1.0, 1.0, alpha],
                alphaMode="BLEND"
            )

        geom.visual.material = mat

def convert_obj_to_glb(obj_path, output_glb_path):
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    alpha_dict = parse_mtl_transparency(mtl_path)

    scene = trimesh.load(obj_path, force='scene')

    apply_transparency_with_materials(scene, alpha_dict)

    scene.export(output_glb_path)
    print("GLB file saved to:", output_glb_path)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    obj_file = os.path.join(base_dir, "output", "brain_with_tumor.obj")
    glb_file = os.path.join(base_dir, "output", "brain_with_tumor.glb")
    convert_obj_to_glb(obj_file, glb_file)
