import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri, colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from SimpleFEM.source.mesh import Mesh


class Config(Enum):
    IMAGES_PATH = 'images'
    COMP_DER_LIMITS = (-20, 0)


class PlotsUtils:

    def __init__(self, mesh: Mesh, penalty: float, elem_volumes: np.ndarray):
        self.mesh = mesh
        self.penalty = penalty
        self.elem_volumes = elem_volumes
        height = mesh.coordinates2D[:,1].max() - mesh.coordinates2D[:,1].min()
        width = mesh.coordinates2D[:,0].max() - mesh.coordinates2D[:,0].min()
        self.ratio = height / width

    def make_plots(
            self,
            displacement: np.ndarray,
            density: np.ndarray,
            comp_derivative: np.ndarray,
            elements_compliance: np.ndarray,
            iteration: int
    ):
        self.draw(
            density,
            os.path.join(Config.IMAGES_PATH.value, f'density/density{iteration}'),
            norm=colors.Normalize(vmin=0, vmax=1),
            cmap='gray_r'
        )
        min_cd, max_cd = Config.COMP_DER_LIMITS.value
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'compliance_derivative/dc{iteration}'),
            norm=colors.Normalize(vmin=min_cd, vmax=max_cd),
            cmap='Blues_r'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'cd_free/dc{iteration}'),
            cmap='Blues_r'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'compliance_derivative_log/dc{iteration}'),
            norm=colors.SymLogNorm(linthresh=1, vmin=min_cd, vmax=max_cd),
            cmap='Blues_r'
        )
        self.draw(
            (density ** self.penalty) * elements_compliance,
            os.path.join(Config.IMAGES_PATH.value, f'compliance/comp{iteration}'),
            cmap='coolwarm'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'power_norm1/dc{iteration}'),
            norm=colors.PowerNorm(gamma=2, vmin=min_cd, vmax=max_cd),
            cmap='Blues_r',
            colorbar_ticks=np.linspace(max_cd, min_cd, 5)
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'power_norm2/dc{iteration}'),
            norm=colors.PowerNorm(gamma=3, vmin=min_cd, vmax=max_cd),
            cmap='Blues_r',
            colorbar_ticks=np.linspace(max_cd, min_cd, 5)
        )
        self.plot_displacements(
            displacements=np.vstack((displacement[:self.mesh.nodes_num], displacement[self.mesh.nodes_num:])).T,
            density=density,
            scale_factor=1,
            file_name=os.path.join(Config.IMAGES_PATH.value, f'displacements/displ{iteration}'),
        )

    def draw(self, elem_values: np.ndarray, file_name: str, norm=None, cmap='gray', colorbar_ticks=None):
        triangulation = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )

        fig, ax = plt.subplots()
        img = ax.tripcolor(triangulation, elem_values, cmap=cmap, norm=norm)

        cbar_ax = ax
        if self.ratio is not None:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * self.ratio)

            ax_div = make_axes_locatable(ax)
            cbar_ax = ax_div.append_axes('right', size='3%', pad='1%')

        fig.colorbar(img, cax=cbar_ax, ticks=colorbar_ticks)

        ax.set_xlabel('x')
        ax.set_ylabel('y', rotation=0)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

    def draw_final_design(self, density: np.ndarray):
        self.draw_final_design_inner(
            density,
            file_name = os.path.join(Config.IMAGES_PATH.value, f'final_design'),
            norm = colors.Normalize(vmin=0, vmax=1),
            cmap = 'gray_r'
        )

    def draw_final_design_inner(self, density: np.ndarray, file_name: str, norm=None, cmap='gray'):
        self.draw(density > 0.5, file_name, norm, cmap)

    def plot_displacements(self, displacements: np.ndarray, density: np.ndarray, scale_factor: float, file_name: str):
        before = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )
        before.set_mask(density < 0.08)
        plt.triplot(before, color='#1f77b4')
        after = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0] + displacements[:, 0] * scale_factor,
            y=self.mesh.coordinates2D[:, 1] + displacements[:, 1] * scale_factor,
            triangles=self.mesh.nodes_of_elem
        )
        after.set_mask(density < 0.08)
        plt.triplot(after, color='#ff7f0e')

        ax = plt.gca()
        if self.ratio is not None:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * self.ratio)

        plt.grid()
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
