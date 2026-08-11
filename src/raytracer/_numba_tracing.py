import numpy as np
from numba import njit, prange


@njit
def _argmax_dot(unit_vectors: np.ndarray, ux: float, uy: float, uz: float):
    """Find the index of the unit vector in unit_vectors that is most parallel to (ux, uy, uv)."""
    best = 0
    best_val = -1e308
    for i in range(unit_vectors.shape[0]):
        v = unit_vectors[i]
        val = v[0] * ux + v[1] * uy + v[2] * uz
        if val > best_val:
            best_val = val
            best = i
    return best


@njit
def _ray_sphere_intersection_single(
    ox: float, oy: float, oz: float, dx: float, dy: float, dz: float, radius: float
) -> tuple[float, float]:
    """Find intersections between a ray and a sphere at origin.

    Solves the equation ||origin + t * direction|| = radius for t.

    Parameters
    ----------
    ox, oy, oz
        Ray origin.
    dx, dy, dz
        Ray direction (assumed normalised).
    radius : float
        Sphere radius.

    Returns
    -------
    t1, t2
        Two intersection parameter values. Invalid intersections are NaN.
    """

    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (ox * dx + oy * dy + oz * dz)
    c = ox * ox + oy * oy + oz * oz - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0.0:
        return np.nan, np.nan
    sd = np.sqrt(disc)
    t1 = (-b - sd) / (2.0 * a)
    t2 = (-b + sd) / (2.0 * a)
    return t1, t2


@njit
def _unique_sorted(arr: np.ndarray, n: int, tol: float) -> int:
    """Sort the first `n` elements of `arr` in-place and remove duplicates.

    After return, the first `k` entries of `arr` (where `k` is the returned
    int) contain the sorted unique values. This avoids returning array slices
    which can confuse Numba's type inference.
    """
    if n == 0:
        return 0

    # simple insertion sort for small n
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    # compact unique values into front of array
    out_count = 1
    last = arr[0]
    for i in range(1, n):
        v = arr[i]
        if np.isfinite(v) and np.abs(v - last) > tol:
            arr[out_count] = v
            last = v
            out_count += 1
    return out_count


@njit
def _compute_t_entry_exit(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    radius: float,
    tol: float,
) -> tuple[float, float, bool]:
    # outer sphere intersection
    t1, t2 = _ray_sphere_intersection_single(ox, oy, oz, dx, dy, dz, radius)
    if not (np.isfinite(t1) and np.isfinite(t2)):
        return np.nan, np.nan, False

    # sort
    t_entry = t1 if t1 <= t2 else t2
    t_exit = t2 if t1 <= t2 else t1

    # Note: `raytracer.regions.SphericalMesh` treats (origin, direction) as an infinite line,
    # so intersections with negative t are still valid here.
    # do not clamp entry: allow negative t_entry for origins inside the sphere

    if np.abs(t_exit - t_entry) <= tol:
        return np.nan, np.nan, False

    return t_entry, t_exit, True


@njit
def _ray_distances_single_fibonacci(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    radial_edges: np.ndarray,
    radius: float,
    n_radial: int,
    unit_vectors: np.ndarray,
    n_samples: int,
    tol: float,
    out: np.ndarray,
    out_row_idx: int,
) -> None:
    # out is a preallocated 2D array shaped (n_rays, n_cells). We write into row
    # `out_row_idx` to avoid passing 1D slices across Numba boundaries.
    n_cells = n_radial * unit_vectors.shape[0]
    # zero output row
    for ii in range(n_cells):
        out[out_row_idx, ii] = 0.0
    t_entry, t_exit, _valid = _compute_t_entry_exit(ox, oy, oz, dx, dy, dz, radius, tol)
    if not _valid:
        return

    # collect candidates
    # avoid Python built-in `max` which can confuse Numba's type inference
    if n_radial - 1 > 0:
        m = n_radial - 1
    else:
        m = 0
    max_cand = 2 + 2 * m + n_samples
    tbuf = np.empty(max_cand, dtype=np.float64)
    count = 0
    tbuf[count] = t_entry
    count += 1
    tbuf[count] = t_exit
    count += 1

    # radial shell intersections
    for ri in range(1, radial_edges.shape[0] - 1):
        r = radial_edges[ri]
        ta, tb = _ray_sphere_intersection_single(ox, oy, oz, dx, dy, dz, r)
        if np.isfinite(ta) and (t_entry + tol) < ta < (t_exit - tol):
            tbuf[count] = ta
            count += 1
        if np.isfinite(tb) and (t_entry + tol) < tb < (t_exit - tol):
            tbuf[count] = tb
            count += 1

    # lateral dense samples (interior)
    # n_samples interior samples evenly spaced between entry and exit
    if n_samples > 0:
        step = (t_exit - t_entry) / (n_samples + 1)
        for si in range(1, n_samples + 1):
            t = t_entry + si * step
            tbuf[count] = t
            count += 1

    new_count = _unique_sorted(tbuf, count, tol)
    if new_count < 2:
        return

    # iterate adjacent segments using tbuf[0:new_count]
    for j in range(new_count - 1):
        t0 = tbuf[j]
        t1_ = tbuf[j + 1]
        if (t1_ - t0) <= tol:
            continue
        tmid = 0.5 * (t0 + t1_)
        px = ox + tmid * dx
        py = oy + tmid * dy
        pz = oz + tmid * dz
        # containment
        if (px * px + py * py + pz * pz) > (radius + tol) * (radius + tol):
            continue
        # radial index
        rnorm = np.sqrt(px * px + py * py + pz * pz)
        if rnorm <= tol:
            radial_index = 0
        else:
            # searchsorted mimic: radial_edges is ascending
            ri_idx = 0
            for rr in range(1, radial_edges.shape[0]):
                if radial_edges[rr] > rnorm:
                    ri_idx = rr - 1
                    break
                if rr == radial_edges.shape[0] - 1:
                    ri_idx = radial_edges.shape[0] - 2
            radial_index = ri_idx

        # lateral index by nearest unit vector
        if rnorm <= tol:
            lateral_index = 0
        else:
            ux = px / rnorm
            uy = py / rnorm
            uz = pz / rnorm
            lateral_index = _argmax_dot(unit_vectors, ux, uy, uz)
        cell_index = radial_index * unit_vectors.shape[0] + lateral_index
        out[out_row_idx, cell_index] += np.abs(t1_ - t0)
    return


@njit(parallel=True)
def ray_distances_batch_fibonacci(
    origins: np.ndarray,
    directions: np.ndarray,
    radial_edges: np.ndarray,
    radius: float,
    n_radial: int,
    unit_vectors: np.ndarray,
    n_samples: int,
    tol: float,
) -> np.ndarray:
    n_rays = origins.shape[0]
    n_cells = n_radial * unit_vectors.shape[0]
    out = np.zeros((n_rays, n_cells), dtype=np.float64)

    for i in prange(n_rays):
        ox = origins[i, 0]
        oy = origins[i, 1]
        oz = origins[i, 2]
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]

        _ray_distances_single_fibonacci(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            radial_edges,
            radius,
            n_radial,
            unit_vectors,
            n_samples,
            tol,
            out,
            i,
        )
    return out
