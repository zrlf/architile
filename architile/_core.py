from typing import Literal, TypeAlias, overload

import numpy as np

from architile.tiling import ArchimedeanTiling

Array: TypeAlias = np.ndarray
Rectangle: TypeAlias = tuple[float, float, float, float]  # (x0, y0, w, h)


@overload
def tile_into_rectangle(
    lx: float,
    ly: float,
    tiling: ArchimedeanTiling,
    theta: float,
    *,
    return_full: Literal[True],
) -> tuple[Array, Array, Array]: ...
@overload
def tile_into_rectangle(
    lx: float,
    ly: float,
    tiling: ArchimedeanTiling,
    theta: float,
    *,
    return_full: Literal[False] = False,
) -> tuple[Array, Array]: ...
def tile_into_rectangle(lx, ly, tiling, theta=0.0, *, return_full=False):
    """Generate a tiling that fits into a rectangle of size (lx, ly).

    Args:
        lx: Length of the rectangle in x direction.
        ly: Length of the rectangle in y direction.
        tiling: Tiling object to use.
        theta: Rotation angle of the tiling (in radians). Defaults to 0.0.

    Returns:
        A tuple (nodes, edges) where nodes is an array of shape (N, 2)
        with the coordinates of the nodes inside the rectangle, edges is an array of
        shape (M, 2) with the indices of the nodes forming each edge
        If `return_full` is True, also returns nodes_all, an array of shape (K, 2) with
        the coordinates of all generated nodes before clipping.
    """
    cell_w = tiling.bravais_vectors()[0, 0]
    cell_h = tiling.bravais_vectors()[1, 1]

    (p0, *_), width, height, _ = spanning_parallelogram(
        tiling.bravais_vectors().T, (0, 0, lx, ly), -theta
    )
    ny = int(np.ceil(height / cell_h))
    nx = int(np.ceil(width / cell_w))

    nodes_all, edges_all = tiling.tile(nx, ny, x0=p0)
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    nodes_all = (R @ nodes_all.T).T

    # clipping
    EPS = 1e-12
    inside = (
        (nodes_all[:, 0] >= -EPS)
        & (nodes_all[:, 0] <= lx + EPS)
        & (nodes_all[:, 1] >= -EPS)
        & (nodes_all[:, 1] <= ly + EPS)
    )
    keep = np.where(inside)[0]

    # renumbering
    new_id = -np.ones(nodes_all.shape[0], dtype=int)
    new_id[keep] = np.arange(keep.size)
    nodes = nodes_all[keep]

    # filter edges
    mask_edges = np.isin(edges_all, keep).all(axis=1)
    edges = edges_all[mask_edges]
    edges = new_id[edges]

    if return_full:
        return nodes, edges, nodes_all
    else:
        return nodes, edges


def spanning_parallelogram(basis: np.ndarray, rectangle: Rectangle, alpha: float):
    """Get the parallelogram spanning a rotated rectangle in a given basis.

    Args:
        basis: 2x2 array with the basis vectors as columns.
        rectangle: (x0, y0, w, h) defining the rectangle in cartesian coordinates.
        alpha: Rotation angle of the rectangle (in radians).

    Returns:
        A tuple (corners, b_width, b_height, rect_rot) where corners is a tuple of the
        four corners of the parallelogram in cartesian coordinates, b_width and b_height
        are the dimensions of the parallelogram in the basis directions, and rect_rot is
        the rotated rectangle coordinates.
    """
    x0, y0, w, h = rectangle
    rect = np.array([[x0, x0 + w, x0 + w, x0], [y0, y0, y0 + h, y0 + h]])
    alpha = -alpha
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    rect_rot = R @ rect

    # basis transform
    b_inv = np.linalg.inv(basis)
    coords_b = b_inv @ rect_rot
    xmin = coords_b.min(axis=1)
    xmax = coords_b.max(axis=1)

    # get parallelogram corners in cartesian coordinates
    p0 = basis @ xmin
    p1 = basis @ np.array([xmax[0], xmin[1]])
    p2 = basis @ xmax
    p3 = basis @ np.array([xmin[0], xmax[1]])
    b_width = (p1 - p0)[0]
    b_height = (p3 - p0)[1]

    return (p0, p1, p2, p3), b_width, b_height, rect_rot
