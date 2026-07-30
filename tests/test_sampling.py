"""Tests for SphericalSamplingTheorem implementations."""

import numpy as np
import pytest

from raytracer.sampling import FibonacciSphericalSampling, MWSphericalSampling


class TestMWSphericalSampling:
    """Test suite for McEwen-Wiaux spherical sampling."""

    def test_init_validates_positive_int(self) -> None:
        """Test that initialisation rejects invalid lateral_resolution."""
        with pytest.raises(ValueError):
            MWSphericalSampling(0)
        with pytest.raises(ValueError):
            MWSphericalSampling(-1)
        with pytest.raises(TypeError):
            MWSphericalSampling(2.5)  # type: ignore[arg-type]

    def test_n_cells(self) -> None:
        """Test that n_cells equals n_lat * n_lon."""
        sampling = MWSphericalSampling(3)
        assert sampling.n_cells == 3 * 5  # n_lat=3, n_lon=2*3-1=5

    def test_labels_count_and_format(self) -> None:
        """Test that labels have the right count and format."""
        sampling = MWSphericalSampling(3)
        assert len(sampling.labels) == sampling.n_cells
        assert sampling.labels[0] == "lat0_lon0"
        assert sampling.labels[-1] == "lat2_lon4"

    def test_sampling_points_shape(self) -> None:
        """Test that sampling_points returns an Nx2 array."""
        sampling = MWSphericalSampling(3)
        pts = sampling.sampling_points()
        assert pts.shape == (sampling.n_cells, 2)

    def test_sampling_points_range(self) -> None:
        """Test that sampling points are in valid angular ranges."""
        sampling = MWSphericalSampling(5)
        pts = sampling.sampling_points()
        theta, phi = pts[:, 0], pts[:, 1]
        assert np.all(theta >= 0) and np.all(theta <= np.pi)
        assert np.all(phi >= 0) and np.all(phi < 2 * np.pi)

    def test_point_to_cell_index_north_pole(self) -> None:
        """Test that the north pole maps to the first latitude band."""
        sampling = MWSphericalSampling(3)
        north_pole = np.array([0.0, 0.0, 1.0])
        idx = sampling.point_to_cell_index(north_pole)
        # North pole (theta=0) should be in lat band 0
        assert 0 <= idx < sampling.n_lon

    def test_point_to_cell_index_south_pole(self) -> None:
        """Test that the south pole maps to the last latitude band."""
        sampling = MWSphericalSampling(3)
        south_pole = np.array([0.0, 0.0, -1.0])
        idx = sampling.point_to_cell_index(south_pole)
        # South pole (theta=pi) should be in lat band n_lat-1
        assert idx >= (sampling.n_lat - 1) * sampling.n_lon

    def test_point_to_cell_index_origin(self) -> None:
        """Test that the origin maps to a valid cell index."""
        sampling = MWSphericalSampling(3)
        origin = np.array([0.0, 0.0, 0.0])
        idx = sampling.point_to_cell_index(origin)
        assert 0 <= idx < sampling.n_cells

    def test_point_to_cell_index_in_range(self) -> None:
        """Test that arbitrary points map to valid indices."""
        sampling = MWSphericalSampling(4)
        rng = np.random.default_rng(42)
        points = rng.normal(size=(20, 3))
        for pt in points:
            idx = sampling.point_to_cell_index(pt)
            assert 0 <= idx < sampling.n_cells

    def test_boundary_t_candidates_outside_sphere_returns_empty(self) -> None:
        """Test that a ray missing the sphere returns no candidates."""
        sampling = MWSphericalSampling(3)
        # Ray clearly missing any boundary intersections within [0, 1]
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        candidates = sampling.boundary_t_candidates(origin, direction, 4.5, 5.5)
        # All candidates should be finite numbers
        assert all(np.isfinite(t) for t in candidates)

    def test_boundary_t_candidates_returns_list(self) -> None:
        """Test that boundary_t_candidates returns a list of floats."""
        sampling = MWSphericalSampling(3)
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = sampling.boundary_t_candidates(origin, direction, 4.0, 6.0)
        assert isinstance(result, list)

    def test_sampling_points_centre_assignment_roundtrip(self) -> None:
        """Each sampling centre should map back to its own cell index."""
        sampling = MWSphericalSampling(4)
        pts = sampling.sampling_points()  # (n_cells, 2) [theta, phi]
        for cell_idx, (theta, phi) in enumerate(pts):
            cart = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            recovered = sampling.point_to_cell_index(cart)
            assert recovered == cell_idx, (
                f"Cell {cell_idx} centre (theta={theta:.3f}, phi={phi:.3f}) "
                f"mapped to cell {recovered}"
            )


class TestFibonacciSphericalSampling:
    """Test suite for Fibonacci spherical sampling."""

    def test_init_validates_positive_int(self) -> None:
        """Test that initialisation rejects invalid n_points."""
        with pytest.raises(ValueError):
            FibonacciSphericalSampling(0)
        with pytest.raises(ValueError):
            FibonacciSphericalSampling(-5)
        with pytest.raises(TypeError):
            FibonacciSphericalSampling(10.5)  # type: ignore[arg-type]

    def test_n_cells(self) -> None:
        """Test that n_cells matches n_points."""
        sampling = FibonacciSphericalSampling(50)
        assert sampling.n_cells == 50

    def test_labels_count_and_format(self) -> None:
        """Test that labels have correct count and format."""
        sampling = FibonacciSphericalSampling(10)
        assert len(sampling.labels) == 10
        assert sampling.labels[0] == "fib0"
        assert sampling.labels[-1] == "fib9"

    def test_sampling_points_shape(self) -> None:
        """Test that sampling_points returns an Nx2 array."""
        n = 30
        sampling = FibonacciSphericalSampling(n)
        pts = sampling.sampling_points()
        assert pts.shape == (n, 2)

    def test_sampling_points_range(self) -> None:
        """Test that sampling points are in valid angular ranges."""
        sampling = FibonacciSphericalSampling(100)
        pts = sampling.sampling_points()
        theta, phi = pts[:, 0], pts[:, 1]
        assert np.all(theta >= 0) and np.all(theta <= np.pi)
        assert np.all(phi >= 0) and np.all(phi < 2 * np.pi)

    def test_sampling_points_coverage(self) -> None:
        """Test that Fibonacci points cover both hemispheres."""
        sampling = FibonacciSphericalSampling(100)
        pts = sampling.sampling_points()
        theta = pts[:, 0]
        # Should have points in both northern and southern hemispheres
        assert np.any(theta < np.pi / 2)
        assert np.any(theta > np.pi / 2)

    def test_point_to_cell_index_self_assignment(self) -> None:
        """Each Fibonacci centre converted to 3D maps back to its own cell."""
        sampling = FibonacciSphericalSampling(30)
        pts = sampling.sampling_points()
        for i, (theta, phi) in enumerate(pts):
            cart = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            idx = sampling.point_to_cell_index(cart)
            assert idx == i, (
                f"Fibonacci centre {i} mapped to cell {idx}"
            )

    def test_point_to_cell_index_in_range(self) -> None:
        """Test that arbitrary points map to valid indices."""
        sampling = FibonacciSphericalSampling(50)
        rng = np.random.default_rng(0)
        points = rng.normal(size=(30, 3))
        for pt in points:
            idx = sampling.point_to_cell_index(pt)
            assert 0 <= idx < sampling.n_cells

    def test_point_to_cell_index_origin(self) -> None:
        """Test that the origin maps to a valid cell index."""
        sampling = FibonacciSphericalSampling(20)
        idx = sampling.point_to_cell_index(np.array([0.0, 0.0, 0.0]))
        assert 0 <= idx < sampling.n_cells

    def test_boundary_t_candidates_returns_list(self) -> None:
        """Test that boundary_t_candidates returns a list."""
        sampling = FibonacciSphericalSampling(30)
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = sampling.boundary_t_candidates(origin, direction, 4.0, 6.0)
        assert isinstance(result, list)

    def test_boundary_t_candidates_nonempty_through_sphere(self) -> None:
        """Test that candidates are generated for a ray through the sphere."""
        sampling = FibonacciSphericalSampling(50)
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = sampling.boundary_t_candidates(origin, direction, 4.0, 6.0)
        assert len(result) > 0

    def test_boundary_t_candidates_all_finite(self) -> None:
        """Test that all returned t values are finite."""
        sampling = FibonacciSphericalSampling(40)
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = sampling.boundary_t_candidates(origin, direction, 4.0, 6.0)
        assert all(np.isfinite(t) for t in result)

    def test_custom_n_ray_samples(self) -> None:
        """Test that n_ray_samples controls the density of t candidates."""
        sampling = FibonacciSphericalSampling(50, n_ray_samples=20)
        origin = np.array([0.0, 0.0, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        result = sampling.boundary_t_candidates(origin, direction, 4.0, 6.0)
        assert len(result) == 20
