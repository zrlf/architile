<div align="center">

<h3 align="center">architile : create lattices with all 11 regular or uniform tilings</h3>

A small python library to generate 2D lattices based on repeating any of the 11 regular or uniform tilings.
Uniform tilings are arrangements of regular polygons that cover a plane without
gaps or overlaps, where each vertex has the same arrangement of polygons.

</div>

## Features

- Get **nodes** and **connectivity** for any of the 11 regular or uniform tilings
- Tile into any rectangular area with options for the boundary (cut, cut
  exactly to bounding box, or fill with partial tiles along the boundary)
- Rotate the tiling pattern by any angle

## All 11 regular or uniform tilings

![architile preview](./examples/tilings.webp)

## Getting started

```py
from architile import tiling, tile_into_rectangle

# Create a SnubSquare tiling
tile = tiling.SnubSquare(a=1.0)

# Tile it into a rectangle of width 5 and height 3 (origin at (0,0))
nodes, edges = tile_into_rectangle((0.0, 0.0, 5.0, 3.0), tile)

# Tile it into a rectangle and add edges along the boundary of the rectangle
# tiling with partial tiles along the bounding box
nodes, edges = tile_into_rectangle((0.0, 0.0, 5.0, 3.0), tile, boundary="cut_fill")

# tile it into a rectangle but put the pattern at an angle of 15 degrees
nodes, edges = tile_into_rectangle((0.0, 0.0, 5.0, 3.0), tile, theta=np.pi/12)
```

## Disclaimer

This is a research project, and the code is provided "as is" without warranty of any kind. Use at your own risk.
