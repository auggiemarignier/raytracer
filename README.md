# Ray Tracer

Pure NumPy implementation for ray tracing through spherical composite geometries.

## Features

- **Geometric regions**: Create spherical shells, hemispheres, and solid balls
- **Composite geometries**: Combine multiple regions (e.g., core + shell)
- **Ray tracing**: Calculate distance travelled through each region
- **Batch operations**: Trace multiple rays simultaneously

## Usage

```python
import numpy as np
from raytracer import SphericalShell, Ball, CompositeRegion, Ray

# Create a composite geometry: inner core + outer core
inner_core = Ball(radius=1221.5)
outer_core = SphericalShell(radius_inner=1221.5, radius_outer=3480.0)

geometry = CompositeRegion(
    [inner_core, outer_core],
    labels=["IC", "OC"],
)

# Define rays (entry and exit points in km)
entry = np.array([[1000.0, 0.0, 0.0]])
exit = np.array([[-800.0, 500.0, 300.0]])

# Calculate distances through each region
ray = Ray(entry, exit)
distances = geometry.ray_distances(ray.origin, ray.direction)
# distances shape: (1,)  total distance through inner_core and outer_core
```

## API

### Regions

- `Ball(radius)` - Solid sphere
- `SphericalShell(radius_inner, radius_outer)` - Spherical shell
- `Hemisphere(radius, normal, centre=None)` - Hemispherical region

### CompositeRegion

- `CompositeRegion(regions, labels=None)` - Combine regions
- `.ray_distances(origin, direction)` - Calculate distances for rays

### Ray

- `Ray(entry, exit)` - Define a ray
- `.origin` - Entry point(s)
- `.direction` - Direction vector(s)
- `.length` - Total ray length(s)
- `.point_at_parameter(t)` - Get point along ray
