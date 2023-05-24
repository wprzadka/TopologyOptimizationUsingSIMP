import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri, colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.plotting_utils import plot_displacements


class Config(Enum):
    IMAGES_PATH = 'images'


class PlotsUtils:

    def __init__(self, mesh: Mesh, penalty: float, elem_volumes: np.ndarray):
        self.mesh = mesh
        self.penalty = penalty
        self.elem_volumes = elem_volumes

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
            ratio=1 / 3,
            norm=colors.Normalize(vmin=0, vmax=1),
            cmap='gray_r'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'compliance_derivative/dc{iteration}'),
            ratio=1 / 3,
            norm=colors.Normalize(vmin=-500, vmax=0),
            cmap='Blues_r'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'cd_free/dc{iteration}'),
            ratio=1 / 3,
            # norm=colors.Normalize(vmin=-1000, vmax=0),
            cmap='Blues_r'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'compliance_derivative_log/dc{iteration}'),
            ratio=1 / 3,
            norm=colors.SymLogNorm(linthresh=1, vmin=-1e3, vmax=0),
            cmap='Blues_r'
        )
        self.draw(
            (density ** self.penalty) * elements_compliance,
            os.path.join(Config.IMAGES_PATH.value, f'compliance/comp{iteration}'),
            ratio=1 / 3,
            cmap='coolwarm'
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'power_norm1/dc{iteration}'),
            ratio=1 / 3,
            norm=colors.PowerNorm(gamma=2, vmin=-1e3, vmax=0),
            cmap='Blues_r',
            colorbar_ticks=[0, -200, -400, -600, -800]
        )
        self.draw(
            comp_derivative,
            os.path.join(Config.IMAGES_PATH.value, f'power_norm2/dc{iteration}'),
            ratio=1 / 3,
            norm=colors.PowerNorm(gamma=3, vmin=-1e3, vmax=0),
            cmap='Blues_r',
            colorbar_ticks=[0, -200, -400, -600, -800]
        )
        plot_displacements(
            displacements=np.vstack((displacement[:self.mesh.nodes_num], displacement[self.mesh.nodes_num:])).T,
            mesh=self.mesh,
            file_name=os.path.join(Config.IMAGES_PATH.value, f'displacements/displ{iteration}')
        )

    def draw(self, elem_values: np.ndarray, file_name: str, ratio: float = None, norm=None, cmap='gray', colorbar_ticks=None):
        triangulation = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )

        fig, ax = plt.subplots()
        img = ax.tripcolor(triangulation, elem_values, cmap=cmap, norm=norm)

        cbar_ax = ax
        if ratio is not None:
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            ax_div = make_axes_locatable(ax)
            cbar_ax = ax_div.append_axes('right', size='3%', pad='1%')

        fig.colorbar(img, cax=cbar_ax, ticks=colorbar_ticks)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

    def draw_final_design(self, density: np.ndarray):
        self.draw_final_design_inner(
            density,
            file_name = os.path.join(Config.IMAGES_PATH.value, f'final_design'),
            ratio = 1 / 3,
            norm = colors.Normalize(vmin=0, vmax=1),
            cmap = 'gray_r'
        )

    def draw_final_design_inner(self, density: np.ndarray, file_name: str, ratio: float = None, norm=None, cmap='gray'):
        self.draw(density > 0.5, file_name, ratio, norm, cmap)
