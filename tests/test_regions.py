"""Test the regions and their ray intersection methods."""

import numpy as np
import pytest

from raytracer import Ball, BallInShell, CompositeRegion, Hemisphere, SphericalShell
from raytracer.regions import _ray_sphere_intersection


class TestRaySphereIntersections:
    """Test suite for ray-sphere intersection calculations."""

    def _test_ray_sphere_intersection(
        self, origins: np.ndarray, directions: np.ndarray, expected: np.ndarray
    ) -> None:
        """Helper method to test ray-sphere intersection calculations."""
        radius = 1.0
        intersections = _ray_sphere_intersection(origins, directions, radius)
        np.testing.assert_allclose(intersections, expected, equal_nan=True)

    def test__ray_sphere_intersections_2_solutions(self) -> None:
        """Test ray-sphere intersections with two intersection points."""
        origins = np.array([[0.0, 0.0, -5.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        expected = np.array([[4.0, 6.0]])

        self._test_ray_sphere_intersection(origins, directions, expected)

    def test__ray_sphere_intersections_one_solution(self) -> None:
        """Test ray-sphere intersections with one intersection point (tangent)."""
        origins = np.array([[1.0, 0.0, -5.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        expected = np.array([[5.0, 5.0]])

        self._test_ray_sphere_intersection(origins, directions, expected)

    def test__ray_sphere_intersections_no_solution(self) -> None:
        """Test ray-sphere intersections with no intersection points."""
        origins = np.array([[2.0, 0.0, -5.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        expected = np.array([[np.nan, np.nan]])

        self._test_ray_sphere_intersection(origins, directions, expected)

    def test__ray_sphere_intersections_origin_on_sphere(self) -> None:
        """Test ray-sphere intersections when the origin is on the sphere surface."""
        origins = np.array([[0.0, 0.0, 1.0]])
        directions = np.array([[0.0, 0.0, 1.0]])
        expected = np.array([[-2.0, 0.0]])

        self._test_ray_sphere_intersection(origins, directions, expected)

    def test__ray_sphere_intersections_multiple_rays(self) -> None:
        """Test ray-sphere intersections for multiple rays."""
        origins = np.array([[0.0, 0.0, -5.0], [2.0, 0.0, -5.0], [1.0, 0.0, -5.0]])
        directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        expected = np.array([[4.0, 6.0], [np.nan, np.nan], [5.0, 5.0]])

        self._test_ray_sphere_intersection(origins, directions, expected)


class TestBall:
    """Test suite for the Ball region."""

    ball = Ball(radius=1.0)

    def test__init__(self) -> None:
        """Test the initialisation of the Ball region."""
        with pytest.raises(ValueError):
            Ball(radius=-1.0)
        assert self.ball.radius == 1.0

    def test_contains(self) -> None:
        """Test the contains method of the Ball region."""

        points_inside = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.9, 0.0, 0.0]])
        points_outside = np.array([[1.5, 0.0, 0.0], [2.0, 2.0, 2.0], [1.1, 0.0, 0.0]])

        assert np.all(self.ball.contains(points_inside))
        assert not np.any(self.ball.contains(points_outside))

    def test_distance_through_ball(self) -> None:
        """Test distance calculation through a Ball region."""

        origins = np.array(
            [
                [0.0, 0.0, -5.0],
                [2.0, 0.0, -5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        directions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        distances = self.ball.ray_distances(origins, directions)

        expected_distances = np.array([2.0, 0.0, 2.0])
        np.testing.assert_allclose(distances, expected_distances)


class TestHemisphere:
    """Test suite for the Hemisphere region."""

    hemisphere = Hemisphere(
        radius=1.0, normal=np.array([0.0, 0.0, 2.0])
    )  # normal need not be unit vector - normalised internally

    def test__init__(self) -> None:
        """Test the initialisation of the Hemisphere region."""
        with pytest.raises(ValueError):
            Hemisphere(radius=-1.0, normal=np.array([0.0, 0.0, 1.0]))
        assert self.hemisphere.radius == 1.0
        np.testing.assert_allclose(self.hemisphere.normal, np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(self.hemisphere.centre, np.array([0.0, 0.0, 0.0]))

    def test_contains(self) -> None:
        """Test the contains method of the Hemisphere region."""

        points_inside = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.9, 0.0, 0.0]])
        points_outside = np.array(
            [
                [1.5, 0.0, 0.0],  # out of radius
                [2.0, 2.0, -2.0],  # out of radius and wrong hemisphere
                [1.1, 0.0, 0.0],  # out of radius
                [0.0, 0.0, -0.1],  # wrong hemisphere
            ]
        )

        assert np.all(self.hemisphere.contains(points_inside))
        assert not np.any(self.hemisphere.contains(points_outside))

    def test_distance_through_hemisphere(self) -> None:
        """Test distance calculation through a Hemisphere region."""

        origins = np.array(
            [
                [0.0, 0.0, -5.0],
                [2.0, 0.0, -5.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.5],
                [1.0, 0.0, -5.0],
            ]
        )
        directions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        distances = self.hemisphere.ray_distances(origins, directions)

        expected_distances = np.array(
            [
                1.0,  # enters through plane, exits sphere (z-axis from below)
                0.0,  # misses hemisphere entirely
                2.0,  # line through centre in-plane
                1.0,  # starts above, exits through plane (z-axis from above)
                np.sqrt(3.0),  # parallel to plane inside hemisphere
                0.0,  # parallel to plane below hemisphere
                0.0,  # tangent at equator and plane point coincide
            ]
        )
        np.testing.assert_allclose(distances, expected_distances)
        # There are lots of awkward cases that need to be added.


class TestSphericalShell:
    """Test suite for the SphericalShell region."""

    shell = SphericalShell(radius_inner=1.0, radius_outer=2.0)

    def test__init__(self) -> None:
        """Test the initialisation of the SphericalShell region."""
        with pytest.raises(ValueError):
            SphericalShell(radius_inner=-1.0, radius_outer=2.0)
        with pytest.raises(ValueError):
            SphericalShell(radius_inner=2.0, radius_outer=1.0)
        assert self.shell.little_ball.radius == 1.0
        assert self.shell.big_ball.radius == 2.0

    def test_contains(self) -> None:
        """Test the contains method of the SphericalShell region."""

        points_inside = np.array([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [-1.5, 0.0, 0.0]])
        points_outside = np.array(
            [
                [0.5, 0.0, 0.0],  # inside inner sphere
                [2.5, 0.0, 0.0],  # outside outer sphere
                [0.0, 0.0, 0.0],  # inside inner sphere
                [3.0, 3.0, 3.0],  # outside outer sphere
            ]
        )

        assert np.all(self.shell.contains(points_inside))
        assert not np.any(self.shell.contains(points_outside))

    def test_distance_through_shell(self) -> None:
        """Test distance calculation through a SphericalShell region."""

        origins = np.array(
            [
                [0.0, 0.0, -5.0],  # through centre
                [3.0, 0.0, 0.0],  # misses outer sphere
                [2.0, 0.0, -5.0],  # tangent to outer sphere
                [1.0, 0.0, -5.0],  # tangent to inner sphere
                [0.0, 1.5, 0.0],  # chord through outer only
                [0.0, 0.0, 0.5],  # offset line through inner + outer
            ]
        )
        directions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        distances = self.shell.ray_distances(origins, directions)

        expected_distances = np.array(
            [
                2.0,  # outer chord (4) minus inner chord (2)
                0.0,  # no intersection with outer sphere
                0.0,  # tangent to outer sphere
                2.0 * np.sqrt(3.0),  # outer chord with inner tangent
                np.sqrt(7.0),  # outer chord only at y=1.5
                np.sqrt(15.0) - np.sqrt(3.0),  # outer minus inner chord at z=0.5
            ]
        )
        np.testing.assert_allclose(distances, expected_distances)


class TestCompositeRegion:
    """Test suite for the CompositeRegion region."""

    composite = CompositeRegion(
        regions=[
            Ball(radius=1.0),
            SphericalShell(radius_inner=1.0, radius_outer=2.0),
        ]
    )
    # This composite is effectively the same as a single Ball of radius 2.0
    # which can be used to verify correctness.
    ball = Ball(radius=2.0)

    def test_contains(self) -> None:
        """Test the contains method of the CompositeRegion region."""

        points_inside = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [-1.9, 0.0, 0.0],
            ]
        )
        points_outside = np.array(
            [
                [2.5, 0.0, 0.0],
                [3.0, 3.0, 3.0],
                [2.1, 0.0, 0.0],
            ]
        )

        assert np.all(self.composite.contains(points_inside))
        assert not np.any(self.composite.contains(points_outside))

    def test_distance_through_composite(self) -> None:
        """Test distance calculation through a CompositeRegion region."""

        origins = np.array(
            [
                [0.0, 0.0, -5.0],
                [3.0, 0.0, -5.0],
                [0.0, 0.0, 0.0],
            ]
        )
        directions = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        distances = self.composite.ray_distances(origins, directions)
        expected_distances = self.ball.ray_distances(origins, directions)
        np.testing.assert_allclose(distances, expected_distances)


class TestBallInShell:
    """Test suite for a CompositeRegion of a Ball inside a SphericalShell."""

    geometry = BallInShell(radius_inner=1.0, radius_outer=2.0)

    def test_labels(self) -> None:
        """Test that the labels are correctly assigned."""
        assert self.geometry.labels == ["ball", "shell"]

    def test_ball_radius_matches_shell_inner_radius(self) -> None:
        """Test that the radius of the ball is the inner radius of the shell."""
        assert self.geometry.ball.radius == self.geometry.shell.radius_inner

    def test_inner_radius_property(self) -> None:
        """Test that the inner_radius property returns the correct value."""
        assert self.geometry.radius_inner == self.geometry.shell.radius_inner

    def test_outer_radius_property(self) -> None:
        """Test that the outer_radius property returns the correct value."""
        assert self.geometry.radius_outer == self.geometry.shell.radius_outer
