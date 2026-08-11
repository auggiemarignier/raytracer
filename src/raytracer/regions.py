"""Geometric regions that compose a sphere."""

from abc import ABC, abstractmethod

import numpy as np

from raytracer.sampling import FibonacciSphericalSampling, SphericalSamplingTheorem

from ._numba_tracing import ray_distances_batch_fibonacci


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
    """A whole-sphere mesh with configurable radial and lateral sampling.

    The lateral partition of the sphere is provided by a
    :class:`~raytracer.sampling.SphericalSamplingTheorem` instance.
    Two built-in options are available:

    * :class:`~raytracer.sampling.MWSphericalSampling` – McEwen-Wiaux
      equiangular grid (exact ray-cell boundary intersections).
    * :class:`~raytracer.sampling.FibonacciSphericalSampling` – Fibonacci
      lattice with nearest-neighbour cell assignment (dense ray sampling).

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    radial_resolution : int
        Number of radial shells between 0 and ``radius``.
    sampling : SphericalSamplingTheorem
        Lateral sampling scheme that defines the angular cells.

    Notes
    -----
    Cell ordering (and therefore ``labels`` and ``ray_distances_per_region``
    columns) is radial-major, then follows the ordering defined by
    ``sampling.labels``.
    """

    def __init__(
        self,
        radius: float,
        radial_resolution: int,
        sampling: SphericalSamplingTheorem,
    ):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if not isinstance(radial_resolution, int) or radial_resolution <= 0:
            raise ValueError("Radial resolution must be a positive integer")
        if not isinstance(sampling, SphericalSamplingTheorem):
            raise TypeError("sampling must be a SphericalSamplingTheorem instance")

        self.radius = radius
        self.radial_resolution = radial_resolution
        self.sampling = sampling

        self.radial_edges = np.linspace(0.0, self.radius, self.radial_resolution + 1)

        self.n_radial = self.radial_resolution
        self.n_cells = self.n_radial * self.sampling.n_cells

        self.labels = [
            f"r{radial}_{lateral_label}"
            for radial in range(self.n_radial)
            for lateral_label in self.sampling.labels
        ]

        self._tolerance = 1e-10

    def contains(self, point: np.ndarray) -> np.ndarray:
        """Check if point(s) are inside the whole sphere."""
        r = np.linalg.norm(point, axis=-1)
        return r <= self.radius

    def ray_distances_per_region(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Calculate distances through each mesh cell separately.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distances : ndarray, shape (n_rays, n_cells)
        """
        origin = np.atleast_2d(np.asarray(origin))
        direction = np.atleast_2d(np.asarray(direction))

        if origin.shape != direction.shape:
            raise ValueError("origin and direction must have matching shapes")
        if origin.shape[-1] != 3:
            raise ValueError("origin and direction must have shape (..., 3)")

        if isinstance(self.sampling, FibonacciSphericalSampling):
            # resolve n_samples to a concrete int (sampling may store None)
            n_samples = (
                self.sampling._n_ray_samples
                if self.sampling._n_ray_samples is not None
                else max(4 * self.sampling.n_cells, 100)
            )

            distances = ray_distances_batch_fibonacci(
                origin,
                direction,
                self.radial_edges,
                self.radius,
                self.n_radial,
                self.sampling._unit_vectors,
                n_samples,
                self._tolerance,
            )
        else:
            n_rays = origin.shape[0]
            distances = np.zeros((n_rays, self.n_cells))

            for i in range(n_rays):
                distances[i] = self._ray_distances_single(origin[i], direction[i])

        return distances

    def ray_distances(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Calculate total distance through the spherical mesh.

        Parameters
        ----------
        origin : ndarray, shape (..., 3) or (3,)
            Ray origin point(s).
        direction : ndarray, shape (..., 3) or (3,)
            Ray direction vector(s) (assumed normalised).

        Returns
        -------
        distances : ndarray, shape (n_rays,)
        """
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

        # Radial shell boundaries
        for radius in self.radial_edges[1:-1]:
            t_inner = _ray_sphere_intersection(
                origin[np.newaxis, :], direction[np.newaxis, :], radius
            )[0]
            t_candidates.extend(
                t
                for t in t_inner
                if np.isfinite(t)
                and t > (t_entry + self._tolerance)
                and t < (t_exit - self._tolerance)
            )

        # Lateral cell boundaries (delegated to the sampling theorem)
        lateral_candidates = self.sampling.boundary_t_candidates(
            origin, direction, t_entry, t_exit
        )
        t_candidates.extend(
            t
            for t in lateral_candidates
            if np.isfinite(t)
            and (t_entry + self._tolerance) < t < (t_exit - self._tolerance)
        )

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
        r = np.linalg.norm(point)

        if r <= self._tolerance:
            radial_index = 0
        else:
            radial_index = int(
                np.clip(
                    np.searchsorted(self.radial_edges, r, side="right") - 1,
                    0,
                    self.n_radial - 1,
                )
            )

        lateral_index = self.sampling.point_to_cell_index(point)

        return radial_index * self.sampling.n_cells + lateral_index


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


def _sorted_unique_parameters(t_values: list[float], tolerance: float) -> np.ndarray:
    """Sort and deduplicate parameter values with a tolerance."""
    sorted_values = np.sort(np.asarray(t_values, dtype=float))
    finite_values = sorted_values[np.isfinite(sorted_values)]
    if finite_values.size == 0:
        return finite_values

    unique_values = [finite_values[0]]
    for value in finite_values[1:]:
        if np.abs(value - unique_values[-1]) > tolerance:
            unique_values.append(value)

    return np.asarray(unique_values)
