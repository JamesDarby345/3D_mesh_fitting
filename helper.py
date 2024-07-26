import numpy as np
import pyvista as pv
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def trimesh_to_pyvista(trimesh_obj):
    """Convert a Trimesh object to a PyVista PolyData object."""
    vertices = trimesh_obj.vertices
    faces = np.column_stack((np.full(len(trimesh_obj.faces), 3), trimesh_obj.faces))
    return pv.PolyData(vertices, faces)

def visualize_data_and_meshes(synthetic_data, optimal_meshes, mesh_only=False):
    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.set_background('white')

    if not mesh_only:
        # Create a grid for the synthetic data
        grid = pv.ImageData()
        grid.dimensions = synthetic_data.shape
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, 1)
        
        # Assign the synthetic data to the grid
        grid.point_data["values"] = synthetic_data.flatten(order="F")
        
        # Add volume to plotter
        plotter.add_volume(grid, cmap="viridis", opacity="linear")

    # Generate a color map for the meshes
    num_meshes = len(optimal_meshes)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_meshes))

    # Add each optimal mesh to the plotter with a unique color
    for i, mesh in enumerate(optimal_meshes):
        color = colors[i][:3]  # RGB values (excluding alpha)
        plotter_mesh = trimesh_to_pyvista(mesh)
        plotter.add_mesh(plotter_mesh, color=color, opacity=1.0 if mesh_only else 0.7)

    # Show the plot
    plotter.show()

def visualize_3d_binary_data(data):
    # Get the shape of the data
    shape = data.shape
    
    # Create coordinates
    x, y, z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Create the 3D visualization
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=data.flatten(),
        isomin=0.5,
        isomax=1,
        opacity=0.5,  # Adjust this value to change the transparency
        surface_count=1,
        colorscale='Viridis',
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, shape[0]], title='X'),
            yaxis=dict(range=[0, shape[1]], title='Y'),
            zaxis=dict(range=[0, shape[2]], title='Z'),
            aspectmode='data'  # Changed to 'data' to respect the actual shape
        ),
        width=700,
        height=700,
        margin=dict(r=20, l=10, b=10, t=10),
        title_text="3D Synthetic Data Visualization (Isosurface)"
    )

    fig.show()

def generate_synthetic_data(size=100, thickness=1, parabola_a=0.01, 
                            parabola_center=(25, 50, 50), 
                            plane_normal=(1, 0, 0), plane_point=(75, 50, 50)):
    """
    Generate synthetic data with a parabola and a plane.
    
    :param size: Size of the 3D array (size x size x size)
    :param thickness: Thickness of the parabola and plane
    :param parabola_a: Coefficient 'a' in the parabola equation y = ax^2
    :param parabola_center: (x, y, z) center point of the parabola
    :param plane_normal: Normal vector of the plane
    :param plane_point: A point on the plane
    :return: 3D NumPy array with the synthetic data
    """
    # Create a 3D array filled with zeros
    data = np.zeros((size, size, size), dtype=np.float32)
    
    # Create coordinate arrays
    x, y, z = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
    
    # Generate parabola
    px, py, pz = parabola_center
    parabola_z = parabola_a * ((x - px)**2 + (y - py)**2) + pz
    parabola_mask = np.abs(z - parabola_z) <= thickness/2
    parabola_zu = -parabola_a * ((x - px)**2 + (y - py)**2) + pz
    parabola_masku = np.abs(z - parabola_zu) <= thickness/2
    data[parabola_mask] = 1
    data[parabola_masku] = 1
    
    # Generate plane
    plane_normal = np.array(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize the normal vector
    d = -np.dot(plane_normal, plane_point)
    plane_eq = plane_normal[0]*x + plane_normal[1]*y + plane_normal[2]*z + d
    plane_mask = np.abs(plane_eq) <= thickness/2
    data[plane_mask] = 1
    
    return data