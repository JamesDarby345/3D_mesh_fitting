from skimage import measure
from scipy.spatial import cKDTree
import trimesh
import numpy as np
from helper import *

def generate_optimal_meshes(volume_data, coverage_threshold=0.9, distance_threshold=5):
    """
    Generate optimal non-intersecting manifold meshes from volumetric data.
    
    :param volume_data: 3D numpy array of binary volumetric data
    :param coverage_threshold: Percentage of non-zero voxels to cover (default: 0.9)
    :param distance_threshold: Maximum distance for a mesh to cover a voxel (default: 5)
    :return: List of trimesh objects representing the optimal meshes
    """
    
    # Step 1: Generate initial meshes
    meshes = generate_initial_meshes(volume_data)
    
    # Step 2: Optimize meshes
    optimized_meshes = optimize_meshes(meshes, volume_data, coverage_threshold, distance_threshold)
    
    return optimized_meshes

def generate_initial_meshes(volume_data):
    """Generate initial meshes using marching cubes algorithm."""
    verts, faces, _, _ = measure.marching_cubes(volume_data, 0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Split the mesh into connected components
    split_meshes = mesh.split(only_watertight=False)

    print("visualizing marching cubes output after cc split") 
    visualize_data_and_meshes(volume_data, split_meshes, mesh_only=True)
    
    # Filter out closed meshes (those that enclose a volume)
    open_meshes = [m for m in split_meshes if not m.is_volume]
    
    return open_meshes

def optimize_meshes(meshes, volume_data, coverage_threshold, distance_threshold):
    """Optimize meshes to meet the given constraints."""
    covered_voxels = np.zeros_like(volume_data, dtype=bool)
    optimized_meshes = []
    
    while np.sum(covered_voxels) / np.sum(volume_data) < coverage_threshold:
        best_mesh = None
        best_coverage = 0
        
        for mesh in meshes:
            # Check if mesh intersects with any optimized mesh
            if any(mesh.intersects_other(m) for m in optimized_meshes):
                continue
            
            # Calculate coverage
            coverage = calculate_coverage(mesh, volume_data, covered_voxels, distance_threshold)
            
            if coverage > best_coverage:
                best_mesh = mesh
                best_coverage = coverage
        
        if best_mesh is None:
            break
        
        optimized_meshes.append(best_mesh)
        covered_voxels |= get_covered_voxels(best_mesh, volume_data, distance_threshold)
        meshes.remove(best_mesh)
    
    return optimized_meshes

def calculate_coverage(mesh, volume_data, covered_voxels, distance_threshold):
    """Calculate the coverage of a mesh."""
    new_covered = get_covered_voxels(mesh, volume_data, distance_threshold)
    return np.sum(new_covered & ~covered_voxels)

def get_covered_voxels(mesh, volume_data, distance_threshold):
    """Get voxels covered by a mesh."""
    voxel_coords = np.array(np.nonzero(volume_data)).T
    mesh_points = mesh.sample(10000)  # Sample points on the mesh
    
    tree = cKDTree(mesh_points)
    distances, _ = tree.query(voxel_coords)
    
    covered = distances <= distance_threshold
    covered_voxels = np.zeros_like(volume_data, dtype=bool)
    covered_voxels[voxel_coords[covered, 0], voxel_coords[covered, 1], voxel_coords[covered, 2]] = True
    
    return covered_voxels