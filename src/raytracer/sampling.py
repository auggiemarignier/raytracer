"""Spherical sampling theorems for use with SphericalMesh.

Each implementation of :class:`SphericalSamplingTheorem` defines a set of
cells on the unit sphere.  A cell is characterised by:

* a centre point (theta, phi) – colatitude and azimuth,
* a rule for deciding which cell a given 3-D Cartesian point belongs to, and
* a method to supply *t*-parameter candidates at lateral cell boundaries
  along a ray, which :class:`~raytracer.regions.SphericalMesh` uses to
  compute per-cell path lengths.
"""

from abc import ABC, abstractmethod

import numpy as np


class SphericalSamplingTheorem(ABC):
    """Abstract base class for spherical sampling theorems.

    Subclasses define a partition of the unit sphere into cells and provide
    the geometry needed for ray-tracing through a
    :class:`~raytracer.regions.SphericalMesh`.
    """

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Number of lateral cells on the sphere."""

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Label for each lateral cell, ordered to match cell indices."""

    @abstractmethod
    def sampling_points(self) -> np.ndarray:
        """Return cell-centre coordinates.

        Returns
        -------
        ndarray, shape (n_cells, 2)
            Each row is ``[theta, phi]`` (colatitude and azimuth) of a cell
            centre, in radians.
        """

    @abstractmethod
    def point_to_cell_index(self, point: np.ndarray) -> int:
        """Map a 3-D Cartesian point to a lateral cell index.

        Parameters
        ----------
        point : ndarray, shape (3,)
            Cartesian coordinates of the query point.

        Returns
        -------
        int
            Index in ``[0, n_cells)``.
        """

    @abstractmethod
    def boundary_t_candidates(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        t_entry: float,
        t_exit: float,
    ) -> list[float]:
        """Return ray-parameter values at lateral cell boundaries.

        These candidates are used by :class:`~raytracer.regions.SphericalMesh`
        to sub-divide a ray path into per-cell segments.  Values outside
        ``(t_entry, t_exit)`` may be returned; the caller is responsible for
        filtering.

        Parameters
        ----------
        origin : ndarray, shape (3,)
            Ray origin.
        direction : ndarray, shape (3,)
            Ray direction (assumed normalised).
        t_entry : float
            Ray parameter at which the ray enters the sphere.
        t_exit : float
            Ray parameter at which the ray exits the sphere.

        Returns
        -------
        list of float
            Candidate *t*-values at lateral cell boundaries.
        """


# ---------------------------------------------------------------------------
# McEwen-Wiaux sampling
# ---------------------------------------------------------------------------


class MWSphericalSampling(SphericalSamplingTheorem):
    """McEwen-Wiaux equiangular sampling on the sphere.

    Uses ``n_lat = L`` latitude bands and ``n_lon = 2L - 1`` longitude
    segments for band-limit ``L`` following the MW equiangular convention.

    Parameters
    ----------
    lateral_resolution : int
        Band-limit ``L`` (positive integer).  Determines the number of
        latitude and longitude cells.
    """

    def __init__(self, lateral_resolution: int) -> None:
        if not isinstance(lateral_resolution, int):
            raise TypeError("lateral_resolution must be an integer")
        if lateral_resolution <= 0:
            raise ValueError("lateral_resolution must be a positive integer")

        self.lateral_resolution = lateral_resolution
        self.n_lat = lateral_resolution
        self.n_lon = 2 * lateral_resolution - 1

        (
            self.theta_centres,
            self.theta_edges,
            self.phi_centres,
            self.phi_step,
            self.phi_offset,
        ) = _mcewen_wiaux_grid(lateral_resolution)

    @property
    def n_cells(self) -> int:
        """Number of lateral cells (n_lat × n_lon)."""
        return self.n_lat * self.n_lon

    @property
    def labels(self) -> list[str]:
        """Labels in row-major order: lat-major, then lon-major."""
        return [
            f"lat{lat}_lon{lon}"
            for lat in range(self.n_lat)
            for lon in range(self.n_lon)
        ]

    def sampling_points(self) -> np.ndarray:
        """Return the MW grid cell centres as an (n_cells, 2) array.

        Returns
        -------
        ndarray, shape (n_cells, 2)
            Columns are ``[theta, phi]`` in radians.
        """
        theta_grid, phi_grid = np.meshgrid(
            self.theta_centres, self.phi_centres, indexing="ij"
        )
        return np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

    def point_to_cell_index(self, point: np.ndarray) -> int:
        """Map a 3-D point to the flat MW lateral cell index.

        Parameters
        ----------
        point : ndarray, shape (3,)
            Cartesian coordinates.

        Returns
        -------
        int
            Flat index in ``[0, n_cells)``.
        """
        _tolerance = 1e-10
        x, y, z = point
        r = np.linalg.norm(point)

        if r <= _tolerance:
            return 0

        theta = np.arccos(np.clip(z / r, -1.0, 1.0))
        phi = np.mod(np.arctan2(y, x), 2 * np.pi)

        lat_index = int(
            np.clip(
                np.searchsorted(self.theta_edges, theta, side="right") - 1,
                0,
                self.n_lat - 1,
            )
        )
        phi_adjusted = np.mod(phi + self.phi_offset, 2 * np.pi)
        lon_index = int(np.clip(int(np.floor(phi_adjusted / self.phi_step)), 0, self.n_lon - 1))

        return lat_index * self.n_lon + lon_index

    def boundary_t_candidates(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        t_entry: float,
        t_exit: float,
    ) -> list[float]:
        """Return exact ray-parameter values at cone and longitude boundaries.

        For MW sampling the lateral cell boundaries are:
        * cones of constant colatitude (theta edges), and
        * half-planes of constant longitude (phi boundaries).

        Parameters
        ----------
        origin, direction, t_entry, t_exit
            See :meth:`SphericalSamplingTheorem.boundary_t_candidates`.

        Returns
        -------
        list of float
        """
        candidates: list[float] = []

        # Cone (colatitude) boundaries
        for theta in self.theta_edges[1:-1]:
            t_theta = _ray_cone_intersections(origin, direction, theta)
            candidates.extend(t for t in t_theta if np.isfinite(t))

        # Longitude (half-plane) boundaries
        phi_boundaries = np.mod(
            np.arange(self.n_lon) * self.phi_step + self.phi_offset, 2 * np.pi
        )
        for phi in phi_boundaries:
            t_phi = _ray_longitude_plane_intersection(origin, direction, phi)
            if np.isfinite(t_phi):
                candidates.append(t_phi)

        return candidates


# ---------------------------------------------------------------------------
# Fibonacci sampling
# ---------------------------------------------------------------------------


class FibonacciSphericalSampling(SphericalSamplingTheorem):
    """Fibonacci lattice sampling on the sphere.

    Points are distributed quasi-uniformly using the golden-angle Fibonacci
    spiral.  Cell boundaries are the Voronoi regions around each Fibonacci
    point; nearest-neighbour (by inner product) determines cell membership.

    Because Fibonacci Voronoi boundaries are not analytically simple,
    ray-tracing uses dense *t*-parameter sampling along the chord to detect
    cell transitions.

    Parameters
    ----------
    n_points : int
        Number of Fibonacci sample points / cells (positive integer).
    n_ray_samples : int, optional
        Number of interior *t* values to generate per ray for boundary
        detection.  Defaults to ``max(4 * n_points, 100)``.
    """

    def __init__(self, n_points: int, n_ray_samples: int | None = None) -> None:
        if not isinstance(n_points, int):
            raise TypeError("n_points must be an integer")
        if n_points <= 0:
            raise ValueError("n_points must be a positive integer")

        self._n_points = n_points
        self._n_ray_samples = n_ray_samples

        # Fibonacci / golden-angle lattice
        i = np.arange(n_points, dtype=float)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        self._theta = np.arccos(np.clip(1.0 - 2.0 * (i + 0.5) / n_points, -1.0, 1.0))
        self._phi = np.mod(i * golden_angle, 2.0 * np.pi)

        # Unit vectors for fast nearest-neighbour lookup
        self._unit_vectors = np.column_stack([
            np.sin(self._theta) * np.cos(self._phi),
            np.sin(self._theta) * np.sin(self._phi),
            np.cos(self._theta),
        ])

    @property
    def n_cells(self) -> int:
        """Number of Fibonacci cells (equal to n_points)."""
        return self._n_points

    @property
    def labels(self) -> list[str]:
        """Label for each Fibonacci cell."""
        return [f"fib{i}" for i in range(self._n_points)]

    def sampling_points(self) -> np.ndarray:
        """Return Fibonacci cell centres as an (n_cells, 2) array.

        Returns
        -------
        ndarray, shape (n_cells, 2)
            Columns are ``[theta, phi]`` in radians.
        """
        return np.stack([self._theta, self._phi], axis=-1)

    def point_to_cell_index(self, point: np.ndarray) -> int:
        """Map a 3-D point to the nearest Fibonacci cell.

        Uses the inner product between the query direction and Fibonacci unit
        vectors; the cell with the largest inner product is returned.

        Parameters
        ----------
        point : ndarray, shape (3,)
            Cartesian coordinates (need not be a unit vector).

        Returns
        -------
        int
            Index of the nearest Fibonacci cell in ``[0, n_cells)``.
        """
        r = np.linalg.norm(point)
        if r < 1e-10:
            return 0
        unit = point / r
        return int(np.argmax(self._unit_vectors @ unit))

    def boundary_t_candidates(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        t_entry: float,
        t_exit: float,
    ) -> list[float]:
        """Return densely-spaced interior *t* values for cell-boundary detection.

        Because Fibonacci Voronoi boundaries have no closed-form ray
        intersection formula, cell transitions are detected by sampling many
        points along the chord.  The density defaults to
        ``max(4 * n_points, 100)`` interior samples.

        Parameters
        ----------
        origin, direction, t_entry, t_exit
            See :meth:`SphericalSamplingTheorem.boundary_t_candidates`.

        Returns
        -------
        list of float
            Interior *t* values (endpoints excluded).
        """
        n = self._n_ray_samples if self._n_ray_samples is not None else max(4 * self._n_points, 100)
        return list(np.linspace(t_entry, t_exit, n + 2)[1:-1])


# ---------------------------------------------------------------------------
# Internal helpers (MW grid geometry)
# ---------------------------------------------------------------------------


def _mcewen_wiaux_grid(
    lateral_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Generate McEwen-Wiaux angular samples and indexing aids.

    Parameters
    ----------
    lateral_resolution : int
        Band-limit ``L``.

    Returns
    -------
    theta_centres : ndarray, shape (L,)
    theta_edges : ndarray, shape (L+1,)
    phi_centres : ndarray, shape (2L-1,)
    phi_step : float
    phi_offset : float
    """
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
    """Find ray-parameter values at intersections with a cone of constant colatitude.

    Returns
    -------
    ndarray, shape (2,)
        Up to two intersection parameters; missing solutions are ``np.nan``.
    """
    x0, y0, z0 = origin
    dx, dy, dz = direction

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos2 = cos_theta**2
    sin2 = sin_theta**2

    a = (dx**2 + dy**2) * cos2 - dz**2 * sin2
    b = 2 * ((x0 * dx + y0 * dy) * cos2 - z0 * dz * sin2)
    c = (x0**2 + y0**2) * cos2 - z0**2 * sin2

    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return np.array([np.nan, np.nan])
        return np.array([-c / b, np.nan])

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return np.array([np.nan, np.nan])

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    return np.array([t1, t2])


def _ray_longitude_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    longitude: float,
) -> float:
    """Find the ray-parameter value at a half-plane of constant longitude."""
    normal = np.array([np.sin(longitude), -np.cos(longitude), 0.0])
    denom = np.dot(normal, direction)
    if np.isclose(denom, 0.0):
        return np.nan
    return float(-np.dot(normal, origin) / denom)
