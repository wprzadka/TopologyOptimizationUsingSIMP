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
    mesh_path = os.path.join(os.path.dirname(__file__), 'meshes/pillar200x300v4.msh')
    mesh = Mesh(mesh_path)

    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['foot-bottom'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['top-left', 'top-right'])

    rhs_func = lambda x: np.array([0, 0])
    dirichlet_func = lambda x: np.array([0, 0])
    neumann_func = lambda x: np.array([0, -1])

    optim = Optimization(
        mesh=mesh,
        material=MaterialProperty.TestMaterial,
        rhs_func=rhs_func,
        dirichlet_func=dirichlet_func,
        neumann_func=neumann_func,
        penalty=3,
        volume_fraction=0.45,
        filter_radius=2.5
    )

    optim.optimize(iteration_limit=100)
