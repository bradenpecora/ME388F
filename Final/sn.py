import time
import concurrent.futures
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


class Material:
    def __init__(self, sigma_t, sigma_s, volumetric_source):
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.sigma_a = sigma_t - sigma_s
        self.volumetric_source = volumetric_source


class Cell:
    def __init__(self, index, xloc, material, length):
        self.id = index
        self.xloc = xloc
        self.material = material
        self.length = length


class Mesh:
    def __init__(self, lengths: list[float], materials: list[Material], dx: float):
        for length in lengths:
            assert length / dx == int(length / dx), "length must be a multiple of dx"
        self.lengths = lengths
        self.total_length = sum(lengths)
        self.dx = dx

        self.cells = []
        idx = 0
        for length, material in zip(lengths, materials):
            for i in range(int(length / dx)):
                self.cells.append(Cell(idx, idx * dx, material, dx))
                idx += 1

        self.n_cells = len(self.cells)

    def get_cell(self, x):
        cell_index = int(x / self.dx)  # floor
        if cell_index == self.n_cells:
            return self.cells[-1]
        return self.cells[cell_index]

    def get_cell_x_positions(self):
        return [cell.xloc + self.dx / 2 for cell in self.cells]

    def get_suface_x_positions(self):
        return [cell.xloc for cell in self.cells] + [self.total_length]


class Material:
    def __init__(self, sigma_t, sigma_s, volumetric_source):
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.sigma_a = sigma_t - sigma_s
        self.volumetric_source = volumetric_source


class Cell:
    def __init__(self, index, xloc, material, length):
        self.id = index
        self.xloc = xloc
        self.material = material
        self.length = length


class Mesh:
    def __init__(self, lengths: list[float], materials: list[Material], dx: float):
        for length in lengths:
            assert length / dx == int(length / dx), "length must be a multiple of dx"
        self.lengths = lengths
        self.total_length = sum(lengths)
        self.dx = dx

        self.cells = []
        idx = 0
        for length, material in zip(lengths, materials):
            for i in range(int(length / dx)):
                self.cells.append(Cell(idx, idx * dx, material, dx))
                idx += 1

        self.n_cells = len(self.cells)

    def get_cell(self, x):
        cell_index = int(x / self.dx)  # floor
        if cell_index == self.n_cells:
            return self.cells[-1]
        return self.cells[cell_index]

    def get_cell_x_positions(self):
        return [cell.xloc + self.dx / 2 for cell in self.cells]

    def get_suface_x_positions(self):
        return [cell.xloc for cell in self.cells] + [self.total_length]


class DiscreteOrdinates:
    def __init__(
        self,
        angles,
        weights,
        mesh,
        left_boundary,
        right_boundary,
        title_start="",
    ):

        self.angles = angles
        self.weights = weights
        self.mesh = mesh
        self.n_cells = mesh.n_cells
        self.n_surfaces = self.n_cells + 1

        assert (
            isinstance(left_boundary, (int, float)) or left_boundary == "reflecting"
        ), "left_boundary must be a float or 'reflecting'"
        self.left_boundary = left_boundary
        assert (
            isinstance(right_boundary, (int, float)) or right_boundary == "reflecting"
        ), "right_boundary must be a float or 'reflecting'"
        self.right_boundary = right_boundary

        self.title_start = title_start

        assert np.all(np.array(angles) != 0), "angles must not be zero"

        self.angular_fluxes_at_surfaces = np.zeros((self.n_surfaces, len(angles)))

        self.average_scalar_flux_at_cells = np.zeros(self.n_cells)

    def angular_flux(self, max_iter=20000, tol=1e-10):

        angular_flux_matrix_objs = {
            angle: AngularFluxMatrix(self, angle) for angle in self.angles
        }
        zero_block = np.zeros((self.n_surfaces, self.n_surfaces))
        number_of_angles = len(self.angles)

        block_matrix_components = (
            []
        )  # this will be a list of lists, where outer list components are rows
        for block_index, angle in enumerate(self.angles):
            submatrix = angular_flux_matrix_objs[angle].A_mat
            row = (
                block_index * [zero_block]
                + [submatrix]
                + (number_of_angles - block_index - 1) * [zero_block]
            )

            block_matrix_components.append(row)

        block_matrix = np.bmat(block_matrix_components)

        # apply neumman boundary condition to boundary
        # this could be better, but homework...
        if self.left_boundary == "reflecting":
            for nth_pos_angle, pos_angle in enumerate(self.angles):
                if pos_angle > 0:
                    neg_angle = -pos_angle
                    nth_neg_angle = self.angles.index(neg_angle)
                    block_matrix[
                        nth_pos_angle * self.n_surfaces, nth_neg_angle * self.n_surfaces
                    ] = -1

        if self.right_boundary == "reflecting":
            for nth_pos_angle, pos_angle in enumerate(self.angles):
                if pos_angle > 0:
                    neg_angle = -pos_angle
                    nth_neg_angle = self.angles.index(neg_angle)
                    block_matrix[
                        nth_neg_angle * self.n_surfaces + self.n_surfaces - 1,
                        nth_pos_angle * self.n_surfaces + self.n_surfaces - 1,
                    ] = -1

        start_time = time.time()
        for iter in range(max_iter):

            # scalar flux at each node

            old_scalar_flux = self.average_scalar_flux_at_cells.copy()

            # generate the b vector based on the old scalar flux
            b_vec = np.concatenate(
                [
                    angular_flux_matrix_objs[angle].b_vec(old_scalar_flux)
                    for angle in self.angles
                ]
            )

            angular_flux_long = np.linalg.solve(block_matrix, b_vec)

            # list where each element is a list of angular fluxes for each angle
            # the same order as self.angles
            self.angular_fluxes_at_surfaces = np.split(
                angular_flux_long, number_of_angles
            )

            self.angular_fluxes_at_cells = [
                angular_flux_matrix_objs[angle].average_angular_flux(
                    self.angular_fluxes_at_surfaces[i]
                )
                for i, angle in enumerate(self.angles)
            ]

            self.average_scalar_flux_at_cells = sum(
                angular_flux_at_angle * weight
                for angular_flux_at_angle, weight in zip(
                    self.angular_fluxes_at_cells, self.weights
                )
            )

            if np.allclose(
                old_scalar_flux, self.average_scalar_flux_at_cells, atol=tol
            ):
                print(f"{self.title_start}: Converged after {iter} iterations")
                end_time = time.time()
                self.solver_time = end_time - start_time
                break

            if iter == max_iter - 1:
                end_time = time.time()
                self.solver_time = end_time - start_time
                print(f"{self.title_start}: Did not converge after {iter} iterations")

        return self.average_scalar_flux_at_cells

    def plot(self):
        fig, ax = plt.subplots()

        ax.plot(
            self.mesh.get_cell_x_positions(),
            self.average_scalar_flux_at_cells,
            label=r"$\langle \phi \rangle$",
            color="purple",
        )

        # for angle, angular_fluxes in zip(self.angles, self.angular_fluxes_at_surfaces):
        #     ax.scatter(
        #         self.mesh.get_suface_x_positions(),
        #         angular_fluxes,
        #         label=rf"$\psi$ surf $\mu=${angle:.2f}",
        #         marker="x",
        #     )

        # for angle, angular_fluxes in zip(self.angles, self.angular_fluxes_at_cells):
        #     ax.scatter(
        #         self.mesh.get_cell_x_positions(),
        #         angular_fluxes,
        #         label=rf"$\langle \psi \rangle$ cell at $\mu=${angle:.2f}",
        #         marker="o",
        #     )

        # ax.set_ylabel(r"$\langle \phi \rangle$")
        ax.set_xlabel("x")
        ax.legend()
        ax.set_title(self.title_start)

        return fig, ax

    def angular_flux_at_xloc(self, xloc):
        # find nearest cell
        cell_index = np.argmin(np.abs(self.cell_x_pos - xloc))

        angular_flux_per_angle = np.array(
            [angular_flux[cell_index] for angular_flux in self.angular_fluxes_at_cells]
        )

        return self.angles, angular_flux_per_angle

    def current(self):
        scalar_flux_derivative = np.gradient(
            self.average_scalar_flux_at_cells, self.cell_x_pos
        )
        transport_xs = self.sigma_t - 0  # since Sigma_s1 = 0
        diffusion_coeff = 1 / 3 / transport_xs
        current = -diffusion_coeff * scalar_flux_derivative
        return self.cell_x_pos, current

    def angular_moments(self, number_of_moments=8):
        legendre_polynomials = [
            sp.special.legendre(i) for i in range(number_of_moments)
        ]

        angular_flux_moments = {}

        for moment_p in range(number_of_moments):
            legendre_polynomial = sp.special.legendre(moment_p)
            angular_flux_moments[moment_p] = sum(
                [
                    weight_m * legendre_polynomial(mu_m) * angular_flux_at_cell
                    for weight_m, mu_m, angular_flux_at_cell in zip(
                        self.weights, self.angles, self.angular_fluxes_at_cells
                    )
                ]
            )

        # reconstruct the angular fluxes from the moments
        angular_fluxes_reconstructed = {}
        for angle in self.angles:
            angular_fluxes_reconstructed[angle] = sum(
                [
                    angular_flux_moments[moment_p]
                    * legendre_polynomials[moment_p](angle)
                    * (2 * moment_p + 1)
                    / 2
                    for moment_p in range(number_of_moments)
                ]
            )

        return angular_flux_moments, angular_fluxes_reconstructed

    def angular_moments_plot(self):
        angular_flux_moments, angular_fluxes_reconstructed = self.angular_moments()

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        for moment_p, angular_flux_moment in angular_flux_moments.items():
            ax[0].plot(self.cell_x_pos, angular_flux_moment, label=f"P{moment_p}")

        ax[0].set_title("Angular Flux Moments")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel(r"$\phi^p$")
        ax[0].legend()

        for angle, angular_flux in angular_fluxes_reconstructed.items():
            ax[1].plot(self.cell_x_pos, angular_flux, label=f"{angle:.2f}")

        ax[1].set_title("Reconstructed Angular Fluxes")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel(r"$\psi(\mu)$")
        ax[1].legend(title=r"$\mu$")

        reconstructed_scalar_flux = sum(
            [
                angular_flux * weight
                for angular_flux, weight in zip(
                    angular_fluxes_reconstructed.values(), self.weights
                )
            ]
        )

        ax[2].plot(self.cell_x_pos, self.average_scalar_flux_at_cells, label="Original")
        ax[2].scatter(self.cell_x_pos, reconstructed_scalar_flux, label="Reconstructed")
        ax[2].set_title("Scalar Flux")
        ax[2].set_xlabel("x")
        ax[2].legend()

        # add text in the center above the plots
        fig.text(0.5, 0.95, self.title_start, ha="center", va="center")

        return fig, ax


class AngularFluxMatrix:
    def __init__(self, discrete_ordinates: DiscreteOrdinates, mu):
        self.mu = mu
        self.discrete_ordinates: DiscreteOrdinates = discrete_ordinates

        self.sign_of_direction = np.sign(mu)
        self.surface_angular_flux = np.zeros(self.discrete_ordinates.n_surfaces)
        self.cell_averaged_angular_flux = np.zeros(self.discrete_ordinates.n_cells)

        self.diag_index = int(-1 * self.sign_of_direction)

        self.A_mat = np.eye(self.discrete_ordinates.n_surfaces)

        for idx in range(self.discrete_ordinates.n_cells):
            cell = self.discrete_ordinates.mesh.cells[idx]
            tau = cell.material.sigma_t * self.sign_of_direction * cell.length / self.mu
            value = -np.exp(-tau)
            if self.sign_of_direction > 0:
                self.A_mat[idx + 1, idx] = value
            elif self.sign_of_direction < 0:
                self.A_mat[idx, idx + 1] = value

    def b_vec(self, scalar_flux_at_cell):

        # self.source_at_cell = [
        #     (cell.material.volumetric_source + scalar_flux * cell.material.sigma_s) / 2
        #     for cell, scalar_flux in zip(self.discrete_ordinates.mesh.cells, scalar_flux_at_cell)
        # ]

        # source_at_cell_with_decay = []
        # for source, cell in zip(self.source_at_cell, self.discrete_ordinates.mesh.cells):
        #     # tau = cell.material.sigma_t * self.sign_of_direction * cell.length / self.mu
        #     tau = abs(cell.material.sigma_t * cell.length / self.mu)
        #     if tau >= 3:
        #         print(f"Warning: tau is {tau} for cell {cell.id}")
        #     source_at_cell_with_decay.append(source / cell.material.sigma_t * (1 - np.exp(-tau)))
        source = []
        for cell_idx, cell in enumerate(self.discrete_ordinates.mesh.cells):
            tau = abs(cell.material.sigma_t * cell.length / self.mu)
            if tau >= 3:
                print(f"Warning: tau is {tau} for cell {cell.id}")
                print(
                    f"sigma_s: {cell.material.sigma_s}, N: {self.discrete_ordinates.mesh.n_cells}, mu: {self.mu}"
                )
            tau_exp = np.exp(-tau)

            fixed_source = cell.material.volumetric_source * (1 - tau_exp) / 2
            scatter_source = (
                scalar_flux_at_cell[cell_idx]
                * cell.material.sigma_s
                * (1 - tau_exp)
                / 2
            )
            source.append((fixed_source + scatter_source) / cell.material.sigma_t)

        if self.sign_of_direction > 0:
            if isinstance(self.discrete_ordinates.left_boundary, (int, float)):
                psi_initial = self.discrete_ordinates.left_boundary
                # I am being lazy with the vacuum boundary condition
                # (not using the extrapolation distance)
            else:
                psi_initial = 0
            return [psi_initial] + source

        elif self.sign_of_direction < 0:
            if isinstance(self.discrete_ordinates.right_boundary, (int, float)):
                psi_initial = self.discrete_ordinates.right_boundary
            else:
                psi_initial = 0
            return source + [psi_initial]

        else:
            raise ValueError("Direction must be positive or negative")

    def average_angular_flux(self, angular_flux_surface):
        # this is really bad code design (dependent on self.source_at_cell/decay_coeff)
        # , but the source at the cell was set up
        # when the b vector was generated...
        # That is, b_vec must have already been run

        # A_coeff = lambda i : self.source_at_cell[i] / self.discrete_ordinates.sigma_t
        old_scalar_flux = self.discrete_ordinates.average_scalar_flux_at_cells

        angular_flux_cell_average = np.zeros(self.discrete_ordinates.n_cells)
        # for i in range(1, self.discrete_ordinates.n_surfaces):
        for cell in self.discrete_ordinates.mesh.cells:
            i = cell.id
            flux_left = angular_flux_surface[i]
            flux_right = angular_flux_surface[i + 1]

            source = (
                cell.material.sigma_s * old_scalar_flux[i]
                + cell.material.volumetric_source
            ) / 2
            A_coeff = source / cell.material.sigma_t

            B_coeff = -self.mu / cell.material.sigma_t / cell.length

            angular_flux_cell_average[i] = A_coeff + B_coeff * (flux_right - flux_left)

        return angular_flux_cell_average
