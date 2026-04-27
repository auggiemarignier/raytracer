"""Ray-geometry intersection and distance calculations."""

import numpy as np

from raytracer.ray import Ray
from raytracer.regions import Region


def calculate_ray_region_distances(
    region: Region,
    ray: Ray,
) -> np.ndarray:
    """Calculate the distance travelled by ray(s) through a region.

    Parameters
    ----------
    region : Region
        The region.
    ray : Ray
        The ray to trace.

    Returns
    -------
    distances : ndarray, shape (...)
        Distance travelled through the region.
    """
    return region.ray_distances(ray.origin, ray.direction)
