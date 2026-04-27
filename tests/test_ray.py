"""Tests for ray operations."""

import numpy as np

from raytracer import Ray


def test_ray_creation():
    """Test basic ray creation."""
    entry = np.array([[1.0, 0.0, 0.0]])
    exit = np.array([[0.0, 1.0, 0.0]])

    ray = Ray(entry, exit)

    np.testing.assert_array_equal(ray.origin, entry)
    assert ray.direction.shape == (1, 3)
    assert ray.length.shape == (1, 1)

    # Direction should be normalised
    np.testing.assert_allclose(
        np.linalg.norm(ray.direction, axis=-1),
        1.0,
        rtol=1e-10,
    )


def test_ray_length():
    """Test ray length calculation."""
    entry = np.array([[0.0, 0.0, 0.0]])
    exit = np.array([[3.0, 4.0, 0.0]])

    ray = Ray(entry, exit)

    expected_length = 5.0
    np.testing.assert_allclose(ray.length, expected_length, rtol=1e-10)


def test_ray_point_at_parameter():
    """Test getting points along the ray."""
    entry = np.array([[0.0, 0.0, 0.0]])
    exit = np.array([[10.0, 0.0, 0.0]])

    ray = Ray(entry, exit)

    # At t=0, should be at origin
    np.testing.assert_array_almost_equal(
        ray.point_at_parameter(0.0),
        entry,
    )

    # At t=1, should be at exit
    np.testing.assert_array_almost_equal(
        ray.point_at_parameter(1.0),
        exit,
    )

    # At t=0.5, should be at midpoint
    np.testing.assert_array_almost_equal(
        ray.point_at_parameter(0.5),
        np.array([[5.0, 0.0, 0.0]]),
    )


def test_ray_batching():
    """Test creating multiple rays at once."""
    entry = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    exit = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    rays = Ray(entry, exit)

    assert rays.origin.shape == (2, 3)
    assert rays.direction.shape == (2, 3)
    assert rays.length.shape == (2, 1)
