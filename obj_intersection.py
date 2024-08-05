import numpy as np
import pyvista as pv
import os
import trimesh
import nrrd

class MeshTranslator:
    def __init__(self, mesh):
        self.mesh = mesh
        self.original_points = mesh.points.copy()
        self.current_position = np.zeros(3)

    def __call__(self, axis, value):
        self.current_position[axis] = value
        self.update()

    def update(self):
        new_points = self.original_points + self.current_position
        self.mesh.points = new_points
        self.mesh.compute_normals()

# Load data
z, y, x = 1744, 2256, 3024
current_dir = os.getcwd()
mesh = trimesh.load_mesh(f'{current_dir}/data/{z:05d}_{y:05d}_{x:05d}/{z}_{y}_{x}_zyx_mesh_15.obj')
mask, header = nrrd.read(f'{current_dir}/data/{z:05d}_{y:05d}_{x:05d}/{z:05d}_{y:05d}_{x:05d}_mask.nrrd')

# Convert trimesh to PyVista mesh
vertices = mesh.vertices
faces = mesh.faces

# Apply transformation to convert vertices from x,y,z to z,y,x and subtract initial values
vertices_transformed = vertices[:, [2, 1, 0]] - np.array([z, y, x])

pv_mesh = pv.PolyData(vertices_transformed, np.column_stack((np.full(len(faces), 3), faces)))

# Create PyVista ImageData from mask
grid = pv.ImageData()
grid.dimensions = np.array(mask.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["values"] = mask.flatten(order="F")

# Create plotter
plotter = pv.Plotter()

# Add orthogonal slices
slices = grid.slice_orthogonal()
volume = plotter.add_mesh(slices, name='volume')

# Add mesh to plotter
mesh_actor = plotter.add_mesh(pv_mesh, color='red', opacity=0.5, label='Mesh')

# Add a legend
plotter.add_legend()

# Initialize MeshTranslator
translator = MeshTranslator(pv_mesh)

# Add sliders for translation
plotter.add_slider_widget(
    callback=lambda value: translator(0, value),
    rng=[-100, 100],
    value=0,
    title="X Translation",
    pointa=(0.025, 0.1),
    pointb=(0.31, 0.1),
    style='modern',
)
plotter.add_slider_widget(
    callback=lambda value: translator(1, value),
    rng=[-100, 100],
    value=0,
    title="Y Translation",
    pointa=(0.025, 0.15),
    pointb=(0.31, 0.15),
    style='modern',
)
plotter.add_slider_widget(
    callback=lambda value: translator(2, value),
    rng=[-100, 100],
    value=0,
    title="Z Translation",
    pointa=(0.025, 0.2),
    pointb=(0.31, 0.2),
    style='modern',
)

# Show the plotter
plotter.show()