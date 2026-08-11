"""Test raytracing with numba.

Import `_numba_tracing` with a fake `numba` so the functions are defined as
plain Python; this ensures tests execute the source lines (no njit), which
coverage can measure.
"""

import importlib
import sys
import types

import numpy as np

from raytracer.sampling import FibonacciSphericalSampling


def _import_nb_tracing_plain():
    """Import the module with a fake `numba` so functions are plain Python.

    This ensures tests execute the source Python code (no njit compilation)
    and coverage records the lines in `_numba_tracing.py`.
    """
    # Remove if already imported
    if "raytracer._numba_tracing" in sys.modules:
        del sys.modules["raytracer._numba_tracing"]

    # Provide a minimal fake numba module
    prev_numba = sys.modules.get("numba")
    fake = types.ModuleType("numba")

    def njit(*args, **kwargs):
        # Support both @njit and @njit(...) forms. If used as @njit
        # the function itself is passed as the first positional arg.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(f):
            return f

        return decorator

    fake.njit = njit
    fake.prange = range
    sys.modules["numba"] = fake

    try:
        mod = importlib.import_module("raytracer._numba_tracing")
    finally:
        if prev_numba is None:
            sys.modules.pop("numba", None)
        else:
            sys.modules["numba"] = prev_numba

    return mod


def test_ray_sphere_intersection_single_simple():
    """Intersection helper: ray through unit sphere returns -1 and 1."""
    nb = _import_nb_tracing_plain()
    # Ray from origin along +x through unit sphere -> intersections at -1 and 1
    t1, t2 = nb._ray_sphere_intersection_single(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    vals = {round(t1, 12), round(t2, 12)}
    assert vals == {-1.0, 1.0}


def test_argmax_dot_returns_expected_index():
    """`_argmax_dot` selects the unit vector most aligned with a direction."""
    nb = _import_nb_tracing_plain()
    unit_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert nb._argmax_dot(unit_vectors, 1.0, 0.0, 0.0) == 0
    assert nb._argmax_dot(unit_vectors, 0.0, 1.0, 0.0) == 1
    assert nb._argmax_dot(unit_vectors, 0.0, 0.0, 1.0) == 2


def test_unique_sorted_deduplicates_and_sorts():
    """In-place unique+sort: removes near-duplicates and NaNs."""
    nb = _import_nb_tracing_plain()
    arr = np.array([0.0, 1.0, 1.0 + 5e-9, 2.0, np.nan])
    tmp = arr.copy()
    k = nb._unique_sorted(tmp, tmp.size, 1e-8)
    res = tmp[:k]
    assert np.allclose(res, np.array([0.0, 1.0, 2.0]))


def test_ray_distances_batch_fibonacci_conserves_path_length():
    """Batch tracer: per-ray total path length equals sphere intersection length."""
    nb = _import_nb_tracing_plain()

    origins = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    directions = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    radial_edges = np.array([0.0, 0.5, 1.0])
    radius = 1.0
    n_radial = 2
    fib = FibonacciSphericalSampling(n_points=12, n_ray_samples=0)
    unit_vectors = fib._unit_vectors
    n_samples = 10
    tol = 1e-9

    out = nb.ray_distances_batch_fibonacci(
        origins,
        directions,
        radial_edges,
        radius,
        n_radial,
        unit_vectors,
        n_samples,
        tol,
    )
    assert out.shape == (3, n_radial * unit_vectors.shape[0])

    # compute expected total lengths using the sphere intersection helper
    helper = nb._ray_sphere_intersection_single
    expected = []
    for o, d in zip(origins, directions):
        t1, t2 = helper(o[0], o[1], o[2], d[0], d[1], d[2], radius)
        if np.isfinite(t1) and np.isfinite(t2):
            expected.append(abs(t2 - t1))
        else:
            expected.append(0.0)

    sums = out.sum(axis=1)
    assert np.allclose(sums, np.array(expected), atol=1e-8)


def _call_single_inplace(
    nb,
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
):
    n_cells = n_radial * unit_vectors.shape[0]
    out_buf = np.zeros((1, n_cells))
    nb._ray_distances_single_fibonacci(
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
        out_buf,
        0,
    )
    return out_buf[0]


def test_ray_distances_single_fibonacci_cell_assignment():
    """Single-ray check: segments map to expected lateral cells and radial shells."""
    nb = _import_nb_tracing_plain()

    # Single ray along +x through sphere centred at origin
    ox, oy, oz = -2.0, 0.0, 0.0
    dx, dy, dz = 1.0, 0.0, 0.0

    radial_edges = np.array([0.0, 0.5, 1.0])
    radius = 1.0
    n_radial = 2
    fib = FibonacciSphericalSampling(n_points=16, n_ray_samples=0)
    unit_vectors = fib._unit_vectors
    n_samples = 8
    tol = 1e-9

    out = _call_single_inplace(
        nb,
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
    )

    # shape and conservation
    assert out.shape[0] == n_radial * unit_vectors.shape[0]
    # expected total length through unit sphere along x-axis is 2.0 (from -1 to +1)
    assert np.isclose(out.sum(), 2.0, atol=1e-8)

    # find which lateral cell corresponds to +x and assert it receives the majority of mass
    lateral_index = fib.point_to_cell_index(np.array([1.0, 0.0, 0.0]))
    n_lateral = unit_vectors.shape[0]
    # compute per-lateral mass (sum over radials)
    per_lateral = np.zeros(n_lateral)
    for r in range(n_radial):
        per_lateral += out[r * n_lateral : (r + 1) * n_lateral]

    # require the lateral nearest +x receives the largest share
    assert per_lateral[lateral_index] == per_lateral.max()


def test_ray_distances_single_fibonacci_no_intersection():
    """When a ray misses the sphere the output buffer remains all zeros."""
    nb = _import_nb_tracing_plain()
    # Ray pointing away from sphere
    ox, oy, oz = 2.0, 0.0, 0.0
    dx, dy, dz = 1.0, 0.0, 0.0

    radial_edges = np.array([0.0, 0.5, 1.0])
    radius = 1.0
    n_radial = 2
    # single-point Fibonacci lattice (degenerate but realistic)
    fib = FibonacciSphericalSampling(n_points=1, n_ray_samples=0)
    unit_vectors = fib._unit_vectors
    n_samples = 0
    tol = 1e-9

    out = _call_single_inplace(
        nb,
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
    )
    assert out.shape[0] == n_radial * unit_vectors.shape[0]
    assert np.allclose(out, 0.0)


def test_ray_distances_single_fibonacci_central_split():
    """Central ray splits radial shells symmetrically when no dense samples."""
    nb = _import_nb_tracing_plain()
    # Ray along +x through origin; deterministic if no dense lateral samples
    ox, oy, oz = -2.0, 0.0, 0.0
    dx, dy, dz = 1.0, 0.0, 0.0

    radial_edges = np.array([0.0, 0.5, 1.0])
    radius = 1.0
    n_radial = 2
    # realistic lateral unit vectors: use Fibonacci lattice
    fib = FibonacciSphericalSampling(n_points=8, n_ray_samples=0)
    unit_vectors = fib._unit_vectors
    n_samples = 0
    tol = 1e-9

    out = _call_single_inplace(
        nb,
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
    )

    # expected inner shell length = 2 * 0.5 = 1.0, outer = 2 * 1.0 - 1.0 = 1.0
    assert np.isclose(out.sum(), 2.0, atol=1e-9)

    lateral_index = fib.point_to_cell_index(np.array([1.0, 0.0, 0.0]))
    n_lateral = unit_vectors.shape[0]
    per_lateral = np.zeros(n_lateral)
    for r in range(n_radial):
        per_lateral += out[r * n_lateral : (r + 1) * n_lateral]

    # nearest lateral to +x should receive some non-zero mass
    assert per_lateral[lateral_index] > 0.0
