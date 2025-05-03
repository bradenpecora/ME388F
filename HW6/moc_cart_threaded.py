import shapely
import numpy as np
import time
import numba

from joblib import Parallel, delayed

MACHINE_EPSILON = 1e-10


class Material:
    def __init__(self, sigma_t, sigma_s, volumetric_source, name=None, color=None):
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.volumetric_source = volumetric_source
        self.name = name
        self.color = color


class CellData:
    def __init__(self, cell: shapely.geometry.Polygon, material: Material):
        self.cell = cell
        self.material = material
        self.area = cell.area
        self.fixed_source = material.volumetric_source * self.area

        self.prior_flux = 0.0
        self.flux = 0.0

    def add_flux(self, flux):
        self.flux += flux / self.area

    def source(self):
        return self.fixed_source + self.prior_flux * self.material.sigma_s

    def reset_flux_return_diff(self):
        difference = self.flux - self.prior_flux
        self.prior_flux = self.flux
        self.flux = 0.0
        return difference


def generate_ray(offset_from_center, line_angle, x_min, x_max, y_min, y_max):
    """Generates a ray an angle `line_angle` that is offset by `offset_from_center` from the center of the problem domain
    in the direction perpendicular to the line angle.
    The ray is bounded by the problem domain.
    Theta is in radians and counterclockwise from the x-axis.
    """
    sin_theta = np.sin(line_angle)
    cos_theta = np.cos(line_angle)
    tan_theta = sin_theta / cos_theta

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # technically not the offset but just a point on the ray we are generating
    x_offset = x_center + offset_from_center * sin_theta
    y_offset = y_center - offset_from_center * cos_theta

    # point intercept form, slope is tan(theta)
    y_of_x = lambda x: tan_theta * (x - x_offset) + y_offset
    x_of_y = lambda y: (y - y_offset) / tan_theta + x_offset

    # edge case where ray intersects corners of the problem domain
    if (
        np.abs(x_max - x_min) == np.abs(y_max - y_min)
        and np.abs(line_angle) == np.pi / 4
        and offset_from_center == 0
    ):
        if line_angle > 0:
            return shapely.geometry.LineString([(x_min, y_min), (x_max, y_max)])
        else:
            return shapely.geometry.LineString([(x_min, y_max), (x_max, y_min)])

    # Calculate the intersection points with the problem domain boundaries
    points = []
    y_of_x_min = y_of_x(x_min)
    if y_min <= y_of_x_min <= y_max:
        points.append((x_min, y_of_x_min))

    y_of_x_max = y_of_x(x_max)
    if y_min <= y_of_x_max <= y_max:
        points.append((x_max, y_of_x_max))

    x_of_y_min = x_of_y(y_min)
    if x_min <= x_of_y_min <= x_max:
        points.append((x_of_y_min, y_min))

    x_of_y_max = x_of_y(y_max)
    if x_min <= x_of_y_max <= x_max:
        points.append((x_of_y_max, y_max))

    if len(points) == 0:
        return None

    assert (
        len(points) == 2
    ), f"Ray does not intersect the problem domain twice. It does {len(points)} times. Points: {points}"

    # Create a line string representing the ray
    ray = shapely.geometry.LineString(points)

    return ray


def normalize_angle(angle):
    """Normalize angle to range [-π, π]"""
    if angle == np.pi:
        return np.pi
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_ray_direction(line_angle):
    """Returns 1 if the ray is traveling from left to right
    and -1 if the ray is traveling from right to left.

    Line angle is in radians
    """
    # Normalize angle to range [-π, π]
    normalized_angle = normalize_angle(line_angle)
    pi_over_2 = np.pi / 2
    if -pi_over_2 < normalized_angle < pi_over_2:
        return 1
    else:
        return -1


def get_first_point(ray, line_angle):
    direction = get_ray_direction(line_angle)

    cord_a, cord_b = ray.coords
    cord_a_x = cord_a[0]
    cord_b_x = cord_b[0]
    if direction == 1:
        return cord_a if cord_a_x < cord_b_x else cord_b
    else:
        return cord_a if cord_a_x > cord_b_x else cord_b


def sort_intersecting_cells_old(ray, angle, cells):
    # Get ray starting point (first coordinate)
    ray_start = get_first_point(ray, angle)

    # Calculate distance from ray start to each cell's intersection point
    cell_distances = []
    for cell in cells:
        intersection = cell.intersection(ray)
        # If the intersection is a point, use that point
        # If it's a line segment, use its midpoint
        if intersection.geom_type == "Point":
            distance_point = intersection
        else:
            distance_point = intersection.centroid

        distance = shapely.distance(shapely.geometry.Point(ray_start), distance_point)
        cell_distances.append((cell, distance))

    # Sort cells by distance from ray start
    cell_distances.sort(key=lambda x: x[1])
    sorted_cells = [cell for cell, _ in cell_distances]
    return sorted_cells


def sort_intersecting_cells(ray, angle, cells, intersections=None):
    """Sort cells by distance along ray.

    Args:
        ray: Ray geometry
        angle: Ray angle
        cells: List of cells
        intersections: Optional pre-computed intersections
    """
    ray_start = get_first_point(ray, angle)
    start_point = shapely.geometry.Point(ray_start)

    distances = [
        (
            shapely.distance(start_point, intersection.centroid)
            if intersection.geom_type != "Point"
            else shapely.distance(start_point, intersection)
        )
        for intersection in intersections
    ]

    sorted_indices = np.argsort(distances)
    return [cells[i] for i in sorted_indices], [
        intersections[i].length for i in sorted_indices
    ]


def get_intersecting_cells(ray, angle, tree: shapely.STRtree):
    # Query the spatial index
    cell_idxs = tree.query(ray, predicate="intersects")
    # Get geometries and calculate intersections in one go
    cells = []
    valid_intersections = []

    cells = tree.geometries.take(cell_idxs)
    intersections = shapely.intersection(cells, ray)
    valid_intersections = [
        intersection for intersection in intersections if intersection.length > 0
    ]

    sorted_cells, sorted_lengths = sort_intersecting_cells(
        ray, angle, cells, valid_intersections
    )

    return sorted_cells, sorted_lengths


class ProductQuadrature:
    def __init__(self, num_azimuthal, num_polar):
        """Azimuthal is planar, over [0, 2pi] space, polar is vertical, over [-pi/2, pi/2] space."""
        self.num_azimuthal = num_azimuthal
        self.num_polar = num_polar

        self.polar_mus, self.polar_weights = np.polynomial.legendre.leggauss(num_polar)
        self.polar_weights = self.polar_weights / 2

        azimuthal_angle_delta = 2 * np.pi / num_azimuthal
        self.azimuthal_angles = (
            np.arange(0, 2 * np.pi, azimuthal_angle_delta) + azimuthal_angle_delta / 2
        )
        self.azimuthal_weights = np.ones(num_azimuthal) * 1 / num_azimuthal

    def azimuthal_angles_weights(self):
        return zip(self.azimuthal_angles, self.azimuthal_weights)

    def polar_mus_weights(self):
        return zip(self.polar_mus, self.polar_weights)


class CartesianMOC:
    def __init__(
        self,
        cell_data_dict: dict[shapely.geometry.Polygon, CellData],
        quadrature: ProductQuadrature,
        bc_north=0,
        bc_east=0,
        bc_south=0,
        bc_west=0,
        ray_width=0.1,
        max_iterations=1000,
    ):
        self.cell_data_dict = cell_data_dict
        self.cell_data_list = list(cell_data_dict.values())
        self.quadrature = quadrature

        self.bc_north = bc_north
        self.bc_east = bc_east
        self.bc_south = bc_south
        self.bc_west = bc_west
        self.ray_width = ray_width
        self.max_iterations = max_iterations

        cells = [cell_data.cell for cell_data in self.cell_data_list]
        self.number_of_cells = len(cells)
        self.tree = shapely.STRtree(cells)

        domain = shapely.union_all(cells)
        self.x_min, self.y_min, self.x_max, self.y_max = domain.bounds
        self.max_ray_offset = (
            max(self.x_max - self.x_min, self.y_max - self.y_min) * 3 / 2
        )  # overestimate

        self.rays_dict = None

    def get_incident_scalar_flux(self, ray, angle):
        first_point = get_first_point(ray, angle)
        first_point_x = first_point[0]
        first_point_y = first_point[1]

        if first_point_x == self.x_min:
            return self.bc_west
        elif first_point_x == self.x_max:
            return self.bc_east
        elif first_point_y == self.y_min:
            return self.bc_south
        elif first_point_y == self.y_max:
            return self.bc_north
        else:
            raise

    def gen_rays(self):
        """
        azimuthal_angle : {
            offset : {
                ray : shapely.geometry.LineString,
                incident_scalar_flux : float,
                intersecting_cells : list[CellData],
                intersection_lengths : list[float],
            }
        }
        """
        print("Generating rays...")
        self.rays_dict = {}
        start_time = time.time()
        for (
            azimuthal_angle,
            azimuthal_weight,
        ) in self.quadrature.azimuthal_angles_weights():
            angle_dict = {}

            offsets = np.arange(
                -self.max_ray_offset, self.max_ray_offset, self.ray_width
            )

            rays = Parallel(n_jobs=-1)(
                delayed(generate_ray)(
                    offset,
                    azimuthal_angle,
                    self.x_min,
                    self.x_max,
                    self.y_min,
                    self.y_max,
                )
                for offset in offsets
            )

            intersecting_rays = [ray for ray in rays if ray is not None]

            intersecting_offsets = [
                offset for offset, ray in zip(offsets, rays) if ray is not None
            ]

            incident_fluxes_per_ray = [
                self.get_incident_scalar_flux(ray, azimuthal_angle)
                for ray in intersecting_rays
            ]

            results = [
                get_intersecting_cells(ray, azimuthal_angle, self.tree)
                for ray in intersecting_rays
            ]
            intersecting_cells_per_ray, intersection_lengths_per_ray = map(
                list, zip(*results)
            )

            intersecting_cell_datas_per_ray = [
                [self.cell_data_dict[cell] for cell in intersecting_cells]
                for intersecting_cells in intersecting_cells_per_ray
            ]

            for (
                offset,
                incident_flux,
                intersecting_cell_datas,
                intersection_lengths,
            ) in zip(
                intersecting_offsets,
                incident_fluxes_per_ray,
                intersecting_cell_datas_per_ray,
                intersection_lengths_per_ray,
            ):
                angle_dict[offset] = {
                    "incident_scalar_flux": incident_flux,
                    "intersecting_cell_datas": intersecting_cell_datas,
                    "intersection_lengths": intersection_lengths,
                    "azimuthal_weight": azimuthal_weight,
                }
            self.rays_dict[azimuthal_angle] = angle_dict
        elapsed_time = time.time() - start_time
        print(f"Rays generated in {elapsed_time:.2f} seconds")
        return self.rays_dict

    def solve_(self, tolerance=1e-6):

        [cell.reset_flux_return_diff() for cell in self.cell_data_list]

        if self.rays_dict == None:
            self.gen_rays()

        for iteration in range(self.max_iterations):
            for azimuthal_angle, angle_dict in self.rays_dict.items():
                for offset, offset_dict in angle_dict.items():
                    if offset_dict is not None:
                        self.solve_ray(**offset_dict)

            differences = [
                cell.reset_flux_return_diff() for cell in self.cell_data_list
            ]
            error = (
                np.sqrt(sum([diff**2 for diff in differences])) / self.number_of_cells
            )

            if np.isnan(error):
                raise
            print(f"Iter {iteration} error: {error:.6f}")
            if error < tolerance:
                print(f"Converged in {iteration} iterations with error {error:.6f}")
                break

    def solve(self, tolerance=1e-6):
        [cell.reset_flux_return_diff() for cell in self.cell_data_list]

        if self.rays_dict is None:
            self.gen_rays()

        ray_args = []
        for azimuthal_angle, angle_dict in self.rays_dict.items():
            for offset, offset_dict in angle_dict.items():
                if offset_dict is not None:
                    ray_args.append(offset_dict)

        for iteration in range(self.max_iterations):
            # Prepare a list of all ray solve arguments
            # Parallelize over rays
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(self.solve_ray)(**offset_dict) for offset_dict in ray_args
            )

            differences = np.array(
                [cell.reset_flux_return_diff() for cell in self.cell_data_list]
            )
            error = np.linalg.norm(differences)

            if np.isnan(error):
                raise
            print(f"Iter {iteration} error: {error:.6f}")
            if error < tolerance:
                print(f"Converged in {iteration} iterations with error {error:.6f}")
                break

    def solve_ray_(
        self,
        incident_scalar_flux,
        intersecting_cell_datas,
        intersection_lengths,
        azimuthal_weight,
    ):
        """Solves ray for all polar angles and adds the flux contribution to the cells based on the azimuthal weight."""
        for polar_mu, polar_weight in self.quadrature.polar_mus_weights():

            incident_angular_flux = (
                incident_scalar_flux * polar_weight * azimuthal_weight
            )

            for intersecting_cell_data, length in zip(
                intersecting_cell_datas, intersection_lengths
            ):
                sigma_t = intersecting_cell_data.material.sigma_t
                mfp = sigma_t * length

                if mfp < MACHINE_EPSILON:
                    exiting_angular_flux = incident_angular_flux
                    average_angular_flux = incident_angular_flux
                else:
                    source = intersecting_cell_data.source()
                    source_over_sigma_t = source / sigma_t
                    tau = mfp / np.abs(polar_mu)
                    exp_term = np.exp(-tau)
                    exiting_angular_flux = (
                        incident_angular_flux * exp_term
                        + source_over_sigma_t * (1 - exp_term)
                    )
                    average_angular_flux = (
                        source_over_sigma_t
                        + (incident_angular_flux - exiting_angular_flux) / tau
                    )
                    assert (
                        np.isfinite(average_angular_flux)
                        and average_angular_flux >= 0.0
                    ), f"Average flux is not finite: {average_angular_flux}"

                flux_contribution = (
                    average_angular_flux
                    * polar_weight
                    * azimuthal_weight
                    * length
                    * self.ray_width
                    / 2
                )
                intersecting_cell_data.add_flux(flux_contribution)

                incident_angular_flux = exiting_angular_flux

    def solve_ray(
        self,
        incident_scalar_flux,
        intersecting_cell_datas,
        intersection_lengths,
        azimuthal_weight,
    ):
        """Vectorized solve for all polar angles and adds the flux contribution to the cells based on the azimuthal weight."""
        for polar_mu, polar_weight in self.quadrature.polar_mus_weights():
            # Extract arrays
            sigma_t = np.array(
                [cell.material.sigma_t for cell in intersecting_cell_datas]
            )
            sources = np.array([cell.source() for cell in intersecting_cell_datas])
            lengths = np.array(intersection_lengths)
            mfp = sigma_t * lengths

            # Avoid division by zero
            mask = mfp >= MACHINE_EPSILON
            tau = np.empty_like(mfp)
            tau[mask] = mfp[mask] / np.abs(polar_mu)
            tau[~mask] = 1.0  # dummy value, won't be used

            exp_term = np.empty_like(mfp)
            exp_term[mask] = np.exp(-tau[mask])
            exp_term[~mask] = 1.0

            source_over_sigma_t = np.empty_like(sources)
            source_over_sigma_t[mask] = sources[mask] / sigma_t[mask]
            source_over_sigma_t[~mask] = 0.0

            # Compute exiting and average angular flux
            exiting_angular_flux = np.empty_like(mfp)
            average_angular_flux = np.empty_like(mfp)

            # Initial angular flux for all segments
            # incident_angular_flux = (
            #     incident_scalar_flux * polar_weight * azimuthal_weight
            # )
            incident_angular_flux = incident_scalar_flux

            # Streaming update: must be serial, but can be vectorized if you don't need streaming
            for i in range(len(mfp)):
                if mask[i]:
                    exiting_angular_flux[i] = incident_angular_flux * exp_term[
                        i
                    ] + source_over_sigma_t[i] * (1 - exp_term[i])
                    average_angular_flux[i] = (
                        source_over_sigma_t[i]
                        + (incident_angular_flux - exiting_angular_flux[i]) / tau[i]
                    )
                    # Optional: check for non-finite/negative
                    assert (
                        np.isfinite(average_angular_flux[i])
                        and average_angular_flux[i] >= 0.0
                    ), f"Average flux is not finite: {average_angular_flux[i]}"
                else:
                    exiting_angular_flux[i] = incident_angular_flux
                    average_angular_flux[i] = incident_angular_flux

                flux_contribution = (
                    average_angular_flux[i]
                    * polar_weight
                    * azimuthal_weight
                    * lengths[i]
                    * self.ray_width
                    / 2
                )
                intersecting_cell_datas[i].add_flux(flux_contribution)
                incident_angular_flux = exiting_angular_flux[i]
