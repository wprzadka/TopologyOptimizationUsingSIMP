import os
import sys

# add SimpleFEM root directory to path in order to make relative imports works
FEM_PATH = os.path.abspath("SimpleFEM")
sys.path.append(FEM_PATH)

from source.mesh import Mesh
from src.optimization import Optimization

if __name__ == '__main__':
    mesh_path = os.path.join(os.path.dirname(__file__), 'meshes/rectangle.msh')
    mesh = Mesh(mesh_path)
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['right'])

    # mesh.draw(True)

    optim = Optimization(mesh=mesh, penalty=3, volume_fraction=0.3)
    optim.optimize()
