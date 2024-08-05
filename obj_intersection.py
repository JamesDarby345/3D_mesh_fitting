import numpy as np
import pyvista as pv
import os
import trimesh
import nrrd
from scipy.spatial import cKDTree

class MeshTranslator:
    def __init__(self, meshes, actors, original_filenames):
        self.meshes = meshes
        self.actors = actors
        self.original_points = [mesh.points.copy() for mesh in meshes]
        self.translations = [np.zeros(3) for _ in meshes]
        self.current_global_translation = np.zeros(3)
        self.original_filenames = original_filenames
        self.mesh_component_map = {}  # Maps original mesh index to list of component indices

    def apply_global_translation(self, axis, value):
        delta = value - self.current_global_translation[axis]
        self.current_global_translation[axis] = value
        for i, actor in enumerate(self.actors):
            if actor.GetVisibility():
                self.translations[i][axis] += delta
                self.update_mesh(i)

    def update_mesh(self, mesh_index):
        mesh = self.meshes[mesh_index]
        new_points = self.original_points[mesh_index] + self.translations[mesh_index]
        mesh.points = new_points
        mesh.compute_normals()

    def get_translation(self, mesh_index):
        return self.translations[mesh_index]

    def get_final_meshes(self):
        final_meshes = []
        for i, mesh in enumerate(self.meshes):
            final_mesh = trimesh.Trimesh(
                vertices=mesh.points,
                faces=mesh.faces.reshape(-1, 4)[:, 1:],
                process=False
            )
            final_meshes.append(final_mesh)
        return final_meshes

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
    filename_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.obj'):
            file_path = os.path.join(folder_path, filename)
            mesh = trimesh.load_mesh(file_path)
            mesh_list.append(mesh)
            filename_list.append(filename)
    return mesh_list, filename_list

class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""
    def __init__(self, actor, translator, mesh_index):
        self.actor = actor
        self.translator = translator
        self.mesh_index = mesh_index

    def __call__(self, state):
        self.actor.SetVisibility(state)
        if state:
            # Apply the mesh's saved translation when made visible
            translation = self.translator.get_translation(self.mesh_index)
            self.translator.update_mesh(self.mesh_index)

# Process and visualize meshes
def process_and_visualize_meshes(folder_path, mask_file, z, y, x, shift_obj):
    # Load meshes
    meshes, original_filenames = load_obj_files(folder_path)
    
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
    mesh_component_map = {}
    
    # Widget size and position
    checkbox_size = 30
    start_pos = 10
    
    for i, mesh in enumerate(meshes):
        # Disconnect non-touching components
        components = disconnect_non_touching_components(mesh)
        mesh_component_map[i] = []
        
        for j, component in enumerate(components):
            vertices = component.vertices
            faces = component.faces
            
            # Apply transformation
            if shift_obj:
                vertices_transformed = vertices[:, [2, 1, 0]] - np.array([z, y, x])
            else:
                vertices_transformed = vertices
            
            pv_mesh = pv.PolyData(vertices_transformed, np.column_stack((np.full(len(faces), 3), faces)))
            pv_meshes.append(pv_mesh)
            mesh_component_map[i].append(len(pv_meshes) - 1)
            
            # Generate a unique color for the mesh
            color = np.random.rand(3)
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Add mesh to plotter with the unique color
            mesh_actor = plotter.add_mesh(pv_mesh, color=color, opacity=0.5, label=f'Mesh {i+1}-{j+1}')
            mesh_actors.append(mesh_actor)
            
            # Set visibility (only first mesh is visible by default)
            is_visible = (i == 0 and j == 0)
            mesh_actor.SetVisibility(is_visible)
    
    # Initialize MeshTranslator for all meshes
    translator = MeshTranslator(pv_meshes, mesh_actors, original_filenames)
    translator.mesh_component_map = mesh_component_map
    
    # Add checkbox for visibility toggle
    for i, (mesh_actor, pv_mesh) in enumerate(zip(mesh_actors, pv_meshes)):
        color = mesh_actor.GetProperty().GetColor()
        color_hex = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        is_visible = mesh_actor.GetVisibility()
        
        callback = SetVisibilityCallback(mesh_actor, translator, i)
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
    
    # Add sliders for global translation (applied to visible meshes)
    plotter.add_slider_widget(
        callback=lambda value: translator.apply_global_translation(0, value),
        rng=[-20, 20],
        value=0,
        title="X Translation",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
    )
    plotter.add_slider_widget(
        callback=lambda value: translator.apply_global_translation(1, value),
        rng=[-20, 20],
        value=0,
        title="Y Translation",
        pointa=(0.025, 0.15),
        pointb=(0.31, 0.15),
        style='modern',
    )
    plotter.add_slider_widget(
        callback=lambda value: translator.apply_global_translation(2, value),
        rng=[-20, 20],
        value=0,
        title="Z Translation",
        pointa=(0.025, 0.2),
        pointb=(0.31, 0.2),
        style='modern',
    )
    
    # Show the plotter
    plotter.show()

    # After closing the plotter, print out the final translations
    print("Final translations:")
    for i, translation in enumerate(translator.translations):
        print(f"Mesh {i+1}: {translation}")

    # Save all meshes (translated or not) with original filenames and component numbers
    final_meshes = translator.get_final_meshes()
    output_folder = os.path.join('output', f'{z:05d}_{y:05d}_{x:05d}')
    os.makedirs(output_folder, exist_ok=True)
    
    for i, original_filename in enumerate(translator.original_filenames):
        components = translator.mesh_component_map[i]
        for j, component_index in enumerate(components):
            mesh = final_meshes[component_index]
            base_name, ext = os.path.splitext(original_filename)
            output_file = os.path.join(output_folder, f"{base_name}_component_{j+1}{ext}")
            mesh.export(output_file)
            print(f"Saved mesh component to: {output_file}")

# Usage
z, y, x = 1744, 2256, 3024
current_directory = os.getcwd()
folder_path = f'{current_directory}/output/{z:05d}_{y:05d}_{x:05d}/'
shift_obj = False
if os.path.isdir(folder_path) is False:
    folder_path = f'{current_directory}/data/{z:05d}_{y:05d}_{x:05d}/'
    shift_obj = True
mask_file = f'{current_directory}/data/{z:05d}_{y:05d}_{x:05d}/{z:05d}_{y:05d}_{x:05d}_mask.nrrd'
process_and_visualize_meshes(folder_path, mask_file, z, y, x, shift_obj)