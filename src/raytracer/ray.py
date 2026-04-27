"""Ray representation and operations."""

import numpy as np


class Ray:
    """A ray defined by entry and exit points through a sphere.

    Parameters
    ----------
    entry : ndarray, shape (3,) or (..., 3)
        Entry point(s) in Cartesian coordinates (x, y, z).
    exit : ndarray, shape (3,) or (..., 3)
        Exit point(s) in Cartesian coordinates (x, y, z).

    Attributes
    ----------
    origin : ndarray
        Ray origin point(s).
    direction : ndarray
        Normalised ray direction vector(s).
    length : ndarray
        Total ray length(s).
    """

    def __init__(self, entry: np.ndarray, exit: np.ndarray):
        self.origin = np.asarray(entry)
        exit = np.asarray(exit)

        # Calculate direction and length
        path_vector = exit - self.origin
        self.length = np.linalg.norm(path_vector, axis=-1, keepdims=True)
        self.direction = path_vector / self.length

    def point_at_parameter(self, t: np.ndarray) -> np.ndarray:
        """Get point(s) along the ray at parameter t.

        Parameters
        ----------
        t : ndarray
            Parameter value(s) where 0 is origin and 1 is exit point.

        Returns
        -------
        ndarray
            Point(s) along the ray.
        """
        return self.origin + t * self.direction * self.length
