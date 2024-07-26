import numpy as np
from helper import *
from generate_meshes import *


# Parameters
size = 50
thickness = 2
parabola_a = 0.02
loc = 25
parabola_center = (25, 25, 25)  # Center of the parabola (x, y, z)
plane_normal = (0, 0, 1)  # Normal vector of the plane
plane_point = (25, 25, 25)  # A point on the plane

# Generate synthetic data
synthetic_data = generate_synthetic_data(size, thickness, parabola_a, 
                                         parabola_center, plane_normal, plane_point)

# Assuming synthetic_data and size are defined
# visualize_3d_binary_data(synthetic_data)

optimal_meshes = generate_optimal_meshes(synthetic_data)
print(optimal_meshes)

visualize_data_and_meshes(synthetic_data, optimal_meshes, mesh_only=True)