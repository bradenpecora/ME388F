import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.interpolate import griddata


def plot_scalar_flux(self, log_color_scale=False):
    fig, ax = plt.subplots()
    values = [cell.prior_flux for cell in self.cell_data_list]

    # Handle zero or negative values if using log scale
    if log_color_scale:
        # min_positive = min([v for v in values if v > 0])
        # # Replace zeros or negative values with a small positive number
        # values = [max(v, min_positive / 10) for v in values]
        values = np.log(values)  # Convert to natural log scale
        label = "ln(Scalar Flux)"
    else:
        label = "Scalar Flux"

    patches = [
        mpatches.Polygon(cell.cell.exterior.coords, closed=True)
        for cell in self.cell_data_list
    ]

    p = PatchCollection(patches, cmap="viridis", alpha=0.8, edgecolors=None)
    p.set_array(np.array(values))
    ax.add_collection(p)

    # Create proper colorbar label based on scale
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label(label)

    ax.set_xlim(self.x_min, self.x_max)
    ax.set_ylim(self.y_min, self.y_max)
    ax.set_aspect("equal")

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")

    return fig


def plot_scalar_flux_3D(self):
    """Creates a smooth 3D surface plot of the scalar flux."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Extract all cell centroids and flux values
    centroids = []
    flux_values = []
    for cell_data in self.cell_data_list:
        centroid = cell_data.cell.centroid
        centroids.append((centroid.x, centroid.y))
        flux_values.append(cell_data.prior_flux)

    # Create a regular grid over the domain
    x_grid = np.linspace(self.x_min, self.x_max, 100)
    y_grid = np.linspace(self.y_min, self.y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate flux values onto the grid using scipy's griddata

    Z = griddata(centroids, flux_values, (X, Y), method="cubic", fill_value=0)

    # Create a smooth surface plot
    surf = ax.plot_surface(
        X, Y, Z, cmap="viridis", edgecolor=None, antialiased=True, alpha=0.8
    )

    # Add a wireframe to better visualize the 3D structure
    ax.plot_wireframe(X, Y, Z, color="black", alpha=0.1, linewidth=0.5)

    # Add the original cell polygons as points for reference
    # centroid_xs = [c[0] for c in centroids]
    # centroid_ys = [c[1] for c in centroids]
    # ax.scatter(centroid_xs, centroid_ys, flux_values, color="red", s=10, alpha=0.5)

    # Set axis labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Flux")
    ax.set_xlim(self.x_min, self.x_max)
    ax.set_ylim(self.y_min, self.y_max)
    ax.set_zlim(0, max(flux_values) * 1.1)

    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label("Scalar Flux")

    plt.title("Smooth 3D Scalar Flux Visualization")
    plt.tight_layout()

    return fig
