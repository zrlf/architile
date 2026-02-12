## v0.2.0 (2026-02-12)

### Feat

- **tiling**: tile periodic with ghost nodes
- **tiling**: wrap connectivity across periodic boundaries (no ghost nodes)
- **ortho**: new triangular pattern with 90 deg periodicity

### Fix

- **tiling**: include module name in tiling name to avoid name conflicts

### Refactor

- **tiling**: make separate tile methods instead of kwarg for periodicity

## v0.1.0 (2026-01-07)

### Feat

- add cut/cut_fill methods for handling boundary of rectangle

### Fix

- cut_fill method lead to duplicate edges in some cases
- add one extra tile in both axis to ensure full tiling of rectangle
