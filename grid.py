# Generate uniform mesh

from dolfin import *
n = 50

# Read mesh
mesh = UnitSquareMesh(n,n)
print('Number of cells = ', mesh.num_cells())
print('Number of vertices =', mesh.num_vertices())

# Save mesh
File('square.xml') << mesh

