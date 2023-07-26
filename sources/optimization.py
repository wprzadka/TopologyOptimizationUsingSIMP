from typing import Callable

import numpy as np

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM
from SimpleFEM.source.utilities.computation_utils import center_of_mass, area_of_triangle
from SimpleFEM.source.examples.materials import MaterialProperty
from sources.plots_utils import PlotsUtils


class Optimization:

    def __init__(
            self,
            mesh: Mesh,
            material: MaterialProperty,
            rhs_func: Callable,
            dirichlet_func: Callable = None,
            neumann_func: Callable = None,
            penalty: float = 3,
            volume_fraction: float = 0.6,
            filter_radius: float = 1.
    ):
        self.mesh = mesh
        self.material = material

        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

        self.penalty = penalty
        self.volume_fraction = volume_fraction
        self.filter_radius = filter_radius

        self.elem_volumes = self.get_elems_volumes()
        self.volume = np.sum(self.elem_volumes)

        self.base_func_ids = [
            np.hstack((nodes_ids, nodes_ids + self.mesh.nodes_num))
            for nodes_ids in self.mesh.nodes_of_elem
        ]
        self.elem_filter_weights = self.get_elements_surrounding()
        self.plots_utils = PlotsUtils(
            mesh=self.mesh,
            penalty=self.penalty,
            elem_volumes=self.elem_volumes
        )

    def bisection(self, x: np.ndarray, comp_deriv: np.ndarray, num_dumping: float = 0.5):
        step = 0.2
        lower = 0
        upper = 1e5

        lower_limit = np.maximum(0.001, x - step)
        upper_limit = np.minimum(1., x + step)

        x_new = None

        while upper - lower > 1e-4:
            mid = lower + (upper - lower) / 2

            # B_e = -(compliance derivative / (lambda * volume derivative))
            beta = (-comp_deriv / (mid * self.elem_volumes)) ** num_dumping
            x_new = np.clip(beta * x, lower_limit, upper_limit)

            # volume [np.sum(self.elem_volumes * x_new)] is monotonously decreasing function of lagrange multiplayer [mid]
            if np.sum(self.elem_volumes * x_new) < self.volume_fraction * self.volume:
                upper = mid
            else:
                lower = mid
        return x_new

    def mesh_independency_filter(self, comp_deriv: np.ndarray, density: np.ndarray):
        neighbours_influence = np.sum((density * comp_deriv) * self.elem_filter_weights, axis=1)
        inertia = density * np.sum(self.elem_filter_weights, axis=1)
        new_comp_deriv = neighbours_influence / inertia
        return new_comp_deriv

    def get_elements_surrounding(self):
        centers = np.array([center_of_mass(self.mesh.coordinates2D[el_nodes]) for el_nodes in self.mesh.nodes_of_elem])
        diffs = centers[:, None] - centers
        distances = np.linalg.norm(diffs, axis=2)
        elem_filter_weights = (self.filter_radius - distances).clip(min=0)
        return elem_filter_weights

    def get_elems_volumes(self):
        volumes = np.array([
            area_of_triangle(self.mesh.coordinates2D[nodes_ids])
            for nodes_ids in self.mesh.nodes_of_elem
        ])
        return volumes

    def compute_elems_compliance(self, density: np.ndarray, displacement: np.ndarray, elem_stiff: list):
        # elements_compliance = np.empty_like(density)
        # for elem_idx in range(elements_compliance.size):
        #     elem_displacement = np.expand_dims(displacement[self.base_func_ids[elem_idx]], 1)
        #     elements_compliance[elem_idx] = elem_displacement.T @ elem_stiff[elem_idx] @ elem_displacement
        # 1335 ms

        elements_compliance = np.squeeze(np.array([
            displacement[None, base_funcs] @ elem_stiff_mat @ displacement[base_funcs, None]
            for base_funcs, elem_stiff_mat in zip(self.base_func_ids, elem_stiff)
        ]))
        # 861 ms
        return elements_compliance

    def optimize(self, iteration_limit: int = 100):

        density = np.full(self.mesh.elems_num, fill_value=self.volume_fraction)

        iteration = 0
        change = 1.

        fem = FEM(
            mesh=self.mesh,
            rhs_func=self.rhs_func,
            dirichlet_func=self.dirichlet_func,
            neumann_func=self.neumann_func,
            young_modulus=self.material.value[0],
            poisson_ratio=self.material.value[1]
        )
        elem_stiff = [fem.construct_local_stiffness_matrix(el_idx) for el_idx in range(self.mesh.elems_num)]

        while change > 1e-4 and iteration < iteration_limit:
            iteration += 1

            displacement = fem.solve(modifier=density ** self.penalty)

            elements_compliance = self.compute_elems_compliance(
                density=density,
                displacement=displacement,
                elem_stiff=elem_stiff
            )

            compliance = np.sum((density ** self.penalty) * elements_compliance)
            comp_derivative = -self.penalty * (density ** (self.penalty - 1)) * elements_compliance
            print(f'iteration: {iteration}')
            print(f'compliance = {compliance}')

            comp_derivative = self.mesh_independency_filter(
                comp_deriv=comp_derivative,
                density=density
            )

            old_density = density.copy()
            density = self.bisection(x=density, comp_deriv=comp_derivative)
            print(f'volume = {np.sum(density * self.elem_volumes)}')
            change = np.max(np.abs(density - old_density))
            print(f'change = {change}')

            if iteration < 25 or iteration % 5 == 0 or iteration in [32, 64]:
                self.plots_utils.make_plots(
                    displacement=displacement,
                    density=density,
                    comp_derivative=comp_derivative,
                    elements_compliance=elements_compliance,
                    iteration=iteration
                )
        self.plots_utils.draw_final_design(density)

