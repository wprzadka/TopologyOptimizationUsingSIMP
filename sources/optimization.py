from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM
from SimpleFEM.source.utilities.computation_utils import center_of_mass, area_of_triangle
from SimpleFEM.source.examples.materials import MaterialProperty


def plot_dispalcements(mesh: Mesh, displacements: np.ndarray, filename: str = 'displacements.png'):
    before = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.triplot(before, color='#1f77b4')
    after = tri.Triangulation(
        x=mesh.coordinates2D[:, 0] + displacements[:, 0],
        y=mesh.coordinates2D[:, 1] + displacements[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.triplot(after, color='#ff7f0e')
    plt.grid()
    plt.savefig(filename)
    plt.close()


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


    def bisection(self, x: np.ndarray, comp_deriv: np.ndarray, numerical_dumping: float = 0.5):
        step = 0.2
        lower = 0
        upper = 1e5

        lower_limit = np.maximum(0.001, x - step)
        upper_limit = np.minimum(1., x + step)

        x_new = None

        while upper - lower > 1e-4:
            mid = lower + (upper - lower) / 2

            beta = (-comp_deriv / mid) ** numerical_dumping
            x_new = np.clip(beta * x, lower_limit, upper_limit)

            # volume [np.sum(self.elem_volumes * x_new)] is monotonously decreasing function of lagrange multiplayer [mid]
            if np.sum(self.elem_volumes * x_new) < self.volume_fraction * self.volume:
                upper = mid
            else:
                lower = mid
        return x_new

    def mesh_independency_filter(self, comp_deriv: np.ndarray, density: np.ndarray):
        new_comp_deriv = np.zeros_like(comp_deriv)
        for el_idx, el_nodes in enumerate(self.mesh.nodes_of_elem):
            el_center = center_of_mass(self.mesh.coordinates2D[el_nodes])

            weights_sum = 0
            combined_sum = 0

            for oth_idx, oth_nodes in enumerate(self.mesh.nodes_of_elem):
                oth_center = center_of_mass(self.mesh.coordinates2D[oth_nodes])
                dist = np.linalg.norm(el_center - oth_center)
                if dist > self.filter_radius:
                    continue
                weight = self.filter_radius - dist
                weights_sum += weight
                combined_sum += weight * density[oth_idx] * comp_deriv[oth_idx]

            new_comp_deriv[el_idx] = combined_sum / (density[el_idx] * weights_sum)
        return new_comp_deriv

    def get_elems_volumes(self):
        volumes = np.array([
            area_of_triangle(self.mesh.coordinates2D[nodes_ids])
            for nodes_ids in self.mesh.nodes_of_elem
        ])
        return volumes

    def draw(self, density: np.ndarray, file_name: str):
        triangulation = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )
        plt.tripcolor(triangulation, density)
        plt.colorbar()
        plt.savefig(file_name)
        plt.close()

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

        while change > 1e-4 and iteration < iteration_limit:
            iteration += 1

            displacement = fem.solve(modifier=density ** self.penalty)

            elements_compliance = np.zeros_like(density)
            for elem_idx, nodes_ids in enumerate(self.mesh.nodes_of_elem):
                base_func_ids = np.hstack((nodes_ids, nodes_ids + self.mesh.nodes_num))
                elem_displacement = np.expand_dims(displacement[base_func_ids], 1)
                elem_stiff = fem.construct_local_stiffness_matrix(elem_idx)

                elements_compliance[elem_idx] = elem_displacement.T @ elem_stiff @ elem_displacement

            compliance = np.sum((density ** self.penalty) * elements_compliance)
            comp_derivative = -self.penalty * (density ** (self.penalty - 1)) * elements_compliance
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

            if iteration % 1 == 0:
                self.draw(density, f'density_{iteration}')
                self.draw(comp_derivative, f'dc_{iteration}')
                self.draw(elements_compliance, f'comp_{iteration}')
                plot_dispalcements(
                    displacements=np.vstack((displacement[:self.mesh.nodes_num], displacement[self.mesh.nodes_num:])).T,
                    mesh=self.mesh,
                    filename=f'displacements_{iteration}'
                )
