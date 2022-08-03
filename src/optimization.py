from enum import Enum

import numpy as np

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM


class MaterialProperty(Enum):
    AluminumAlloys = (10.2, 0.33)
    BerylliumCopper = (18.0, 0.29)
    CarbonSteel = (29.0, 0.29)
    CastIron = (14.5, 0.21)


class Optimization:

    def __init__(
            self,
            mesh: Mesh,
            penalty: float,
            volume_fraction: float,
    ):
        self.mesh = mesh
        self.penalty = penalty
        self.volume_fraction = volume_fraction

    def bisection(self, x: np.ndarray, comp_deriv: np.ndarray):
        step = 0.1
        lower = 0
        upper = 100

        while lower < upper:
            mid = lower + (upper - lower) / 2

            lower_limit = np.maximum(0, x - step)
            upper_limit = np.minimum(1, x + step)

            beta = (comp_deriv / mid) ** self.penalty
            x_new = np.clip(beta * x, lower_limit, upper_limit)
            if np.sum(x_new) < self.volume_fraction * self.mesh.nodes_num:
                upper = mid
            else:
                lower = mid

    # def clamp(self, x, lower, upper):
    #     return max(min(x, upper), lower)

    def optimize(self):

        iteration_limit = 1
        # TODO density per element (not per node)
        density = np.full(self.mesh.nodes_num, fill_value=self.volume_fraction)

        iteration = 0
        change = 1.
        while change > 0.01 and iteration < iteration_limit:
            iteration += 1

            rhs_func = lambda x: np.array([0, 0])
            dirichlet_func = lambda x: np.array([0, 0])
            neumann_func = lambda x: np.array([-0, -1])

            fem = FEM(
                mesh=self.mesh,
                rhs_func=rhs_func,
                dirichlet_func=dirichlet_func,
                neumann_func=neumann_func,
                young_modulus=MaterialProperty.CarbonSteel.value[0],
                poisson_ratio=MaterialProperty.CarbonSteel.value[1]
            )
            # TODO density has to be added to stiffness matrix
            displacement = fem.solve(density=density)

            compliance = displacement.T @ fem.stiffness_matrix @ displacement
            comp_derivative = -self.penalty * (compliance ** (self.penalty - 1))
            self.bisection(x=displacement, comp_deriv=comp_derivative)

            print(displacement)
