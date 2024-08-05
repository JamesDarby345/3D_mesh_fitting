import numpy as np
import pyvista as pv
import os
import trimesh
import nrrd
from scipy.spatial import cKDTree

class MeshTranslator:
    def __init__(self, meshes):
        self.meshes = meshes
        self.original_points = [mesh.points.copy() for mesh in meshes]
        self.current_position = np.zeros(3)

    def __call__(self, axis, value):
        self.current_position[axis] = value
        self.update()

    def update(self):
        for i, mesh in enumerate(self.meshes):
            new_points = self.original_points[i] + self.current_position
            mesh.points = new_points
            mesh.compute_normals()

def disconnect_non_touching_components(mesh, distance_threshold=0.1):
    """Separate non-touching components of a mesh."""
    # Use trimesh to get connected components
    components = mesh.split(only_watertight=False)
    
    # Function to check if two components are touching
    def are_touching(comp1, comp2):
        tree = cKDTree(comp1.vertices)
        distances, _ = tree.query(comp2.vertices, k=1)
        return np.any(distances < distance_threshold)
    
    # Group touching components
    grouped_components = []
    for comp in components:
        added = False
        for group in grouped_components:
            if any(are_touching(comp, existing_comp) for existing_comp in group):
                group.append(comp)
                added = True
                break
        if not added:
            grouped_components.append([comp])
    
    # Merge components in each group
    return [trimesh.util.concatenate(group) for group in grouped_components]

# Load all .obj files from a folder
def load_obj_files(folder_path):
    mesh_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.obj'):
            file_path = os.path.join(folder_path, filename)
            mesh = trimesh.load_mesh(file_path)
            mesh_list.append(mesh)
    return mesh_list

class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""
    def __init__(self, actor):
        self.actor = actor

    def __call__(self, state):
        self.actor.SetVisibility(state)

# Process and visualize meshes
def process_and_visualize_meshes(folder_path, mask_file, z, y, x):
    # Load meshes
    meshes = load_obj_files(folder_path)
    
    # Load mask
    mask, header = nrrd.read(mask_file)
    
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
    
    # Process each mesh
    pv_meshes = []
    mesh_actors = []
    
    # Widget size and position
    checkbox_size = 30
    start_pos = 10
    
    for i, mesh in enumerate(meshes):
        # Disconnect non-touching components
        components = disconnect_non_touching_components(mesh)
        
        for j, component in enumerate(components):
            vertices = component.vertices
            faces = component.faces
            
            # Apply transformation
            vertices_transformed = vertices[:, [2, 1, 0]] - np.array([z, y, x])
            
            pv_mesh = pv.PolyData(vertices_transformed, np.column_stack((np.full(len(faces), 3), faces)))
            pv_meshes.append(pv_mesh)
            
            # Generate a unique color for the mesh
            color = np.random.rand(3)
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Add mesh to plotter with the unique color
            mesh_actor = plotter.add_mesh(pv_mesh, color=color, opacity=0.5, label=f'Mesh {i+1}-{j+1}')
            mesh_actors.append(mesh_actor)
            
            # Set visibility (only first mesh is visible by default)
            is_visible = (i == 0 and j == 0)
            mesh_actor.SetVisibility(is_visible)
            
            # Add checkbox for visibility toggle
            callback = SetVisibilityCallback(mesh_actor)
            plotter.add_checkbox_button_widget(
                callback,
                value=is_visible,
                position=(10, start_pos),
                size=checkbox_size,
                border_size=1,
                color_on=color_hex,
                color_off='grey',
                background_color='grey',
            )
            start_pos += checkbox_size + 5
    
    # Add a legend
    plotter.add_legend()
    
    # Initialize MeshTranslator for all meshes
    translator = MeshTranslator(pv_meshes)
    
    # Add sliders for translation (applied to all meshes)
    plotter.add_slider_widget(
        callback=lambda value: translator(0, value),
        rng=[-20, 20],
        value=0,
        title="X Translation",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
    )
    plotter.add_slider_widget(
        callback=lambda value: translator(1, value),
        rng=[-20, 20],
        value=0,
        title="Y Translation",
        pointa=(0.025, 0.15),
        pointb=(0.31, 0.15),
        style='modern',
    )
    plotter.add_slider_widget(
        callback=lambda value: translator(2, value),
        rng=[-20, 20],
        value=0,
        title="Z Translation",
        pointa=(0.025, 0.2),
        pointb=(0.31, 0.2),
        style='modern',
    )
    
    # Show the plotter
    plotter.show()

# Usage
z, y, x = 1744, 2256, 3024
folder_path = '/Users/jamesdarby/Documents/VesuviusScroll/GP/3D_Mesh_Fitting/data/01744_02256_03024/'
mask_file = '/Users/jamesdarby/Documents/VesuviusScroll/GP/3D_Mesh_Fitting/data/01744_02256_03024/01744_02256_03024_mask.nrrd'
process_and_visualize_meshes(folder_path, mask_file, z, y, x)