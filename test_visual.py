import trimesh
import pyrender
import numpy as np
from PIL import Image

# Create a colored material
material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.2, 0.5, 1.0, 1.0],  # light blue
    metallicFactor=0.3,
    roughnessFactor=0.7
)

# Create mesh and assign material
mesh = trimesh.creation.icosphere(subdivisions=3)
mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

# Build scene
scene = pyrender.Scene()
scene.add(mesh)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
scene.add(light, pose=np.eye(4))
scene.add(camera, pose=[[1,0,0,0],[0,1,0,0],[0,0,1,2.5],[0,0,0,1]])

# Render
r = pyrender.OffscreenRenderer(256, 256)
color, _ = r.render(scene)
r.delete()
Image.fromarray(color).save("colored_sphere.png")
print("✅ saved colored_sphere.png")