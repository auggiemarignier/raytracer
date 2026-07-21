"""Ray tracing through 3D spherical geometries."""

from raytracer.intersection import calculate_ray_region_distances
from raytracer.ray import Ray
from raytracer.regions import (
    Ball,
    BallInShell,
    CompositeRegion,
    Hemisphere,
    Region,
    SphericalMesh,
    SphericalShell,
)

__all__ = [
    "Ray",
    "Region",
    "SphericalShell",
    "Hemisphere",
    "Ball",
    "BallInShell",
    "CompositeRegion",
    "SphericalMesh",
    "calculate_ray_region_distances",
]
