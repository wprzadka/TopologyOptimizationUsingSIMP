import os
import sys

import numpy as np

# add SimpleFEM root directory to path in order to make relative imports works
FEM_PATH = os.path.abspath("SimpleFEM")
sys.path.append(FEM_PATH)

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.examples.materials import MaterialProperty
from sources.optimization import Optimization

if __name__ == '__main__':
    mesh_path = os.path.join(os.path.dirname(__file__), 'meshes/truss_domain.msh')
    mesh = Mesh(mesh_path)
    mesh.draw()
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left-down', 'right-down'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['mid-up'])

    rhs_func = lambda x: np.array([0, -9.81])
    dirichlet_func = lambda x: np.array([0, 0])
    neumann_func = lambda x: np.array([0, -1e6])

    optim = Optimization(
        mesh=mesh,
        material=MaterialProperty.Polystyrene,
        rhs_func=rhs_func,
        dirichlet_func=dirichlet_func,
        neumann_func=neumann_func,
        penalty=3,
        volume_fraction=0.5,
        filter_radius=8.
    )
    optim.optimize(iteration_limit=50)
