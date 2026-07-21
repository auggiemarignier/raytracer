"""Geometric regions that compose a sphere."""

from abc import ABC, abstractmethod

import numpy as np


class Region(ABC):
    """Base class for geometric regions within a sphere.

    Each region defines a bounded volume and can compute the distance
    travelled by a ray through it.
    """

    @abstractmethod
    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if point(s) are inside the region.

        Parameters
        ----------
        point : ndarray, shape (..., 3)
            Point(s) in Cartesian coordinates.

        Returns
        -------
        ndarray, shape (...)
            Boolean array indicating membership.
        """
        pass

    @abstractmethod
    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance(s) travelled by ray(s) through the region.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distance : ndarray, shape (...,)
            Distance travelled through region. Zero if no intersection.
        """
        pass


class Ball(Region):
    """A solid sphere (ball).

    Parameters
    ----------
    radius : float
        Radius of the ball.
    """

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if points are within the ball."""
        r = np.linalg.norm(point, axis=-1)
        return r <= self.radius

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance through the ball.

        Returns the distance from the first intersection to the second.
        """
        origin = np.asarray(origin)
        direction = np.asarray(direction)

        t_intersections = _ray_sphere_intersection(origin, direction, self.radius)

        # Take the two intersection points
        t1 = t_intersections[..., 0]
        t2 = t_intersections[..., 1]

        # Distance is the difference
        distances = np.where(
            np.isfinite(t1) & np.isfinite(t2),
            np.abs(t2 - t1),
            0.0,  # No intersection yields zero distance
        )

        return distances


class SphericalShell(Region):
    """A spherical shell (region between two concentric spheres).

    Parameters
    ----------
    radius_inner : float
        Inner radius of the shell.
    radius_outer : float
        Outer radius of the shell.
    """

    def __init__(self, radius_inner: float, radius_outer: float):
        if radius_inner >= radius_outer:
            raise ValueError("radius_inner must be less than radius_outer")
        if radius_inner < 0 or radius_outer <= 0:
            raise ValueError("Radii must be positive")
        self.little_ball = Ball(radius_inner)
        self.big_ball = Ball(radius_outer)

    @property
    def radius_inner(self) -> float:
        """Inner radius of the shell (read-only)."""
        return self.little_ball.radius

    @property
    def radius_outer(self) -> float:
        """Outer radius of the shell (read-only)."""
        return self.big_ball.radius

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if points are within the shell."""
        return self.big_ball.contains(point) & ~self.little_ball.contains(point)

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance through the shell."""
        return self.big_ball.ray_distances(
            origin, direction
        ) - self.little_ball.ray_distances(origin, direction)


class Hemisphere(Region):
    """A hemispherical region (half of a sphere along a plane).

    Parameters
    ----------
    radius : float
        Radius of the hemisphere.
    normal : ndarray, shape (3,)
        Normal vector of the dividing plane (points towards the positive hemisphere).
    centre : ndarray, shape (3,), optional
        Centre of the hemisphere. Default is origin.
    """

    def __init__(
        self,
        radius: float,
        normal: np.ndarray,
        centre: np.ndarray | None = None,
    ):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius
        if (norm := np.linalg.norm(normal)) == 0:
            raise ValueError("Normal vector cannot be zero")
        self.normal = np.asarray(normal) / norm
        self.centre = np.asarray(centre) if centre is not None else np.zeros(3)

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if points are within the hemisphere."""
        point = np.asarray(point)

        # Check if within the sphere
        r = np.linalg.norm(point - self.centre, axis=-1)
        in_sphere = r <= self.radius

        # Check if on the correct side of the plane
        relative_pos = point - self.centre
        side = np.sum(relative_pos * self.normal, axis=-1) >= 0

        return in_sphere & side

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distance through the hemisphere.

        A ray intersects the hemisphere boundary at up to three points:
        - Two on the spherical surface
        - One on the dividing plane

        Of these, only 2 will be valid intersection points lying on the hemisphere.
        The distance is the absolute difference between the smallest and largest valid t values.
        If fewer than two valid points exist, the distance is zero.
        """
        origin = np.atleast_2d(origin)
        direction = np.atleast_2d(direction)

        # Ray-sphere intersections (up to 2 points)
        t_sphere = _ray_sphere_intersection(
            origin - self.centre, direction, self.radius
        )
        sphere_points = origin[:, None, :] + t_sphere[..., None] * direction[:, None, :]
        sphere_valid = self.contains(sphere_points) & np.isfinite(t_sphere)

        # Ray-plane intersection (up to 1 point)
        denom = np.sum(direction * self.normal, axis=-1)
        ray_parallel_to_plane = np.abs(denom) < 1e-8
        t_plane = np.full(origin.shape[0], np.nan)
        t_plane[~ray_parallel_to_plane] = (
            np.sum(
                (self.centre - origin[~ray_parallel_to_plane]) * self.normal, axis=-1
            )
            / denom[~ray_parallel_to_plane]
        )

        plane_points = origin + t_plane[:, None] * direction
        plane_valid = (
            (~ray_parallel_to_plane)
            & np.isfinite(t_plane)
            & self.contains(plane_points)
        )

        # Collect valid t values per ray
        t1 = np.where(sphere_valid[:, 0], t_sphere[:, 0], np.nan)
        t2 = np.where(sphere_valid[:, 1], t_sphere[:, 1], np.nan)
        tp = np.where(plane_valid, t_plane, np.nan)

        candidates = np.stack([t1, t2, tp], axis=-1)  # (n_rays, 3)

        valid_counts = np.sum(np.isfinite(candidates), axis=-1)
        t_min = np.nanmin(candidates, axis=-1)
        t_max = np.nanmax(candidates, axis=-1)

        distances = np.where(valid_counts >= 2, np.abs(t_max - t_min), 0.0)

        return distances.squeeze()


class CompositeRegion(Region):
    """A composition of multiple regions forming a complete geometry.

    Parameters
    ----------
    regions : list of Region
        List of regions, in order.
    labels : list of str, optional
        Labels for each region.
    """

    # TODO: Something to validate that the regions form a complete sphere with no gaps or overlaps

    def __init__(self, regions: list[Region], labels: list[str] | None = None):
        self.regions = regions
        self.labels = (
            labels
            if labels is not None
            else [f"region_{i}" for i in range(len(regions))]
        )

        if len(self.labels) != len(self.regions):
            raise ValueError("Number of labels must match number of regions")

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if a point is within any of the regions."""
        return np.any([region.contains(point) for region in self.regions], axis=0)

    def ray_distances_per_region(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Calculate distances through each region separately.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distances : ndarray, shape (..., n_regions)
            Distances through each region.
        """
        origin = np.atleast_2d(origin)
        direction = np.atleast_2d(direction)

        n_rays = origin.shape[0]

        distances = np.zeros((n_rays, len(self.regions)))

        for i, region in enumerate(self.regions):
            distances[:, i] = region.ray_distances(origin, direction)

        return distances

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate distances through all regions.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distances : ndarray, shape (...,)
            Distance through the whole region.
        """
        distances = self.ray_distances_per_region(origin, direction)

        return distances.sum(axis=1)


class SphericalMesh(Region):
    """A whole-sphere mesh with radial and McEwen-Wiaux lateral sampling.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    radial_resolution : int
        Number of radial cells between 0 and ``radius``.
    lateral_resolution : int
        McEwen-Wiaux band-limit for lateral sampling.
        The mesh uses ``n_lat = lateral_resolution`` latitude cells and
        ``n_lon = 2 * lateral_resolution - 1`` longitude cells.

    Notes
    -----
    Cell ordering (and therefore ``labels`` and ``ray_distances_per_region`` columns)
    is radial-major, then latitude-major, then longitude-major.
    """

    def __init__(
        self,
        radius: float,
        radial_resolution: int,
        lateral_resolution: int,
    ):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if not isinstance(radial_resolution, int) or radial_resolution <= 0:
            raise ValueError("radial_resolution must be a positive integer")
        if not isinstance(lateral_resolution, int) or lateral_resolution <= 0:
            raise ValueError("lateral_resolution must be a positive integer")

        self.radius = radius
        self.radial_resolution = radial_resolution
        self.lateral_resolution = lateral_resolution

        self.radial_edges = np.linspace(0.0, self.radius, self.radial_resolution + 1)
        (
            self.theta_centres,
            self.theta_edges,
            self.phi_centres,
            self.phi_step,
            self.phi_offset,
        ) = _mcewen_wiaux_grid(self.lateral_resolution)

        self.n_radial = self.radial_resolution
        self.n_lat = self.lateral_resolution
        self.n_lon = 2 * self.lateral_resolution - 1
        self.n_cells = self.n_radial * self.n_lat * self.n_lon

        self.labels = [
            f"r{radial}_lat{lat}_lon{lon}"
            for radial in range(self.n_radial)
            for lat in range(self.n_lat)
            for lon in range(self.n_lon)
        ]

        self._tolerance = 1e-10

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if point(s) are inside the whole sphere."""
        r = np.linalg.norm(point, axis=-1)
        return r <= self.radius

    def ray_distances_per_region(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Calculate distances through each mesh cell separately."""
        origin = np.atleast_2d(np.asarray(origin))
        direction = np.atleast_2d(np.asarray(direction))

        if origin.shape != direction.shape:
            raise ValueError("origin and direction must have matching shapes")
        if origin.shape[-1] != 3:
            raise ValueError("origin and direction must have shape (..., 3)")

        n_rays = origin.shape[0]
        distances = np.zeros((n_rays, self.n_cells))

        for i in range(n_rays):
            distances[i] = self._ray_distances_single(origin[i], direction[i])

        return distances

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate total distance through the spherical mesh."""
        distances = self.ray_distances_per_region(origin, direction)
        return distances.sum(axis=1)

    def _ray_distances_single(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Calculate per-cell distances for a single ray."""
        distances = np.zeros(self.n_cells)

        t_outer = _ray_sphere_intersection(
            origin[np.newaxis, :], direction[np.newaxis, :], self.radius
        )[0]
        if not np.all(np.isfinite(t_outer)):
            return distances

        t_entry, t_exit = np.sort(t_outer)
        if np.abs(t_exit - t_entry) <= self._tolerance:
            return distances

        t_candidates = [t_entry, t_exit]

        for radius in self.radial_edges[1:-1]:
            t_inner = _ray_sphere_intersection(
                origin[np.newaxis, :], direction[np.newaxis, :], radius
            )[0]
            t_candidates.extend(
                t
                for t in t_inner
                if np.isfinite(t) and (t_entry + self._tolerance) < t < (t_exit - self._tolerance)
            )

        for theta in self.theta_edges[1:-1]:
            t_theta = _ray_cone_intersections(origin, direction, theta)
            t_candidates.extend(
                t
                for t in t_theta
                if np.isfinite(t) and (t_entry + self._tolerance) < t < (t_exit - self._tolerance)
            )

        phi_boundaries = (np.arange(self.n_lon) + 0.5) * self.phi_step
        for phi in phi_boundaries:
            t_phi = _ray_longitude_plane_intersection(origin, direction, phi)
            if (
                np.isfinite(t_phi)
                and (t_entry + self._tolerance) < t_phi < (t_exit - self._tolerance)
            ):
                t_candidates.append(t_phi)

        t_sorted = _sorted_unique_parameters(t_candidates, self._tolerance)

        for t0, t1 in zip(t_sorted[:-1], t_sorted[1:], strict=False):
            if (t1 - t0) <= self._tolerance:
                continue

            t_mid = 0.5 * (t0 + t1)
            point = origin + t_mid * direction
            if not self.contains(point):
                continue

            cell_index = self._point_to_cell_index(point)
            distances[cell_index] += np.abs(t1 - t0)

        return distances

    def _point_to_cell_index(self, point: np.ndarray) -> int:
        """Map a Cartesian point to a flat mesh-cell index."""
        x, y, z = point
        r = np.linalg.norm(point)

        if r <= self._tolerance:
            theta = 0.0
            phi = 0.0
            radial_index = 0
        else:
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.mod(np.arctan2(y, x), 2 * np.pi)
            radial_index = np.searchsorted(self.radial_edges, r, side="right") - 1
            radial_index = int(np.clip(radial_index, 0, self.n_radial - 1))

        lat_index = np.searchsorted(self.theta_edges, theta, side="right") - 1
        lat_index = int(np.clip(lat_index, 0, self.n_lat - 1))

        phi_shifted = np.mod(phi + self.phi_offset, 2 * np.pi)
        lon_index = int(np.floor(phi_shifted / self.phi_step))
        lon_index = int(np.clip(lon_index, 0, self.n_lon - 1))

        return (radial_index * self.n_lat + lat_index) * self.n_lon + lon_index


class BallInShell(CompositeRegion):
    """A composite region consisting of a ball inside a spherical shell.

    Parameters
    ----------
    radius_inner : float
        Inner radius of the shell (also the radius of the ball).
    radius_outer : float
        Outer radius of the shell.
    """

    def __init__(self, radius_inner: float, radius_outer: float):
        self.ball = Ball(radius_inner)
        self.shell = SphericalShell(radius_inner, radius_outer)
        super().__init__(regions=[self.ball, self.shell], labels=["ball", "shell"])

    @property
    def radius_inner(self) -> float:
        """Inner radius of the shell (read-only)."""
        return self.shell.radius_inner

    @property
    def radius_outer(self) -> float:
        """Outer radius of the shell (read-only)."""
        return self.shell.radius_outer


def _ray_sphere_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Find intersections between a ray and a sphere at origin.

    Solves the equation ||origin + t * direction|| = radius for t.

    Parameters
    ----------
    origin : ndarray, shape (..., 3)
        Ray origin(s).
    direction : ndarray, shape (..., 3)
        Ray direction(s) (assumed normalised).
    radius : float
        Sphere radius.

    Returns
    -------
    t : ndarray, shape (..., 2)
        Two intersection parameter values. Invalid intersections are NaN.
    """
    origin = np.asarray(origin)
    direction = np.asarray(direction)

    # Ray: P(t) = origin + t * direction
    # Sphere: ||P||^2 = radius^2
    # Substituting: ||origin + t*direction||^2 = radius^2
    # Expanding: ||origin||^2 + 2*t*(origin·direction) + t^2*||direction||^2 = radius^2

    # Coefficients of quadratic equation: a*t^2 + b*t + c = 0
    a = np.sum(direction * direction, axis=-1)
    b = 2.0 * np.sum(origin * direction, axis=-1)
    c = np.sum(origin * origin, axis=-1) - radius**2

    discriminant = b**2 - 4 * a * c

    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))  # force at least one solution
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t1 = np.where(discriminant >= 0, t1, np.nan)  # nan when no solution
    t2 = np.where(discriminant >= 0, t2, np.nan)  # nan when no solution

    # The smallest t in absolute value is the intersection closest to the origin
    # t1 is the most negative.
    return np.stack([t1, t2], axis=-1)


def _mcewen_wiaux_grid(
    lateral_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Generate McEwen-Wiaux latitude/longitude sample locations and indexing aids."""
    n_lat = lateral_resolution
    n_lon = 2 * lateral_resolution - 1

    theta_centres = np.pi * (2 * np.arange(n_lat) + 1) / (2 * n_lat - 1)
    theta_edges = np.empty(n_lat + 1)
    theta_edges[0] = 0.0
    theta_edges[-1] = np.pi
    if n_lat > 1:
        theta_edges[1:-1] = 0.5 * (theta_centres[:-1] + theta_centres[1:])

    phi_step = 2 * np.pi / n_lon
    phi_centres = phi_step * np.arange(n_lon)
    phi_offset = 0.5 * phi_step

    return theta_centres, theta_edges, phi_centres, phi_step, phi_offset


def _ray_cone_intersections(
    origin: np.ndarray,
    direction: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Find intersections between a line and a cone of constant colatitude."""
    x0, y0, z0 = origin
    dx, dy, dz = direction

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos2 = cos_theta**2
    sin2 = sin_theta**2

    a = (dx**2 + dy**2) * cos2 - (dz**2) * sin2
    b = 2 * ((x0 * dx + y0 * dy) * cos2 - (z0 * dz) * sin2)
    c = (x0**2 + y0**2) * cos2 - (z0**2) * sin2

    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return np.array([np.nan, np.nan])
        return np.array([-c / b, np.nan])

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return np.array([np.nan, np.nan])

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    return np.array([t1, t2])


def _ray_longitude_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    longitude: float,
) -> float:
    """Find intersection between a line and a plane at constant longitude."""
    normal = np.array([np.sin(longitude), -np.cos(longitude), 0.0])
    denom = np.dot(normal, direction)
    if np.isclose(denom, 0.0):
        return np.nan
    return -np.dot(normal, origin) / denom


def _sorted_unique_parameters(t_values: list[float], tolerance: float) -> np.ndarray:
    """Sort and deduplicate parameter values with a tolerance."""
    values = np.sort(np.asarray(t_values, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values

    unique_values = [values[0]]
    for value in values[1:]:
        if np.abs(value - unique_values[-1]) > tolerance:
            unique_values.append(value)

    return np.asarray(unique_values)
