from enum import Enum
from typing import Literal, TypeAlias, overload

import numpy as np

from architile.tiling import ArchimedeanTiling

Array: TypeAlias = np.ndarray
Rectangle: TypeAlias = tuple[float, float, float, float]  # (x0, y0, w, h)


class BoundaryHandling(Enum):
    CUT = "cut"
    DISCARD = "discard"
    CUT_FILL = "cut_fill"


EPS = 1e-12


@overload
def tile_into_rectangle(
    rect: Rectangle,
    tiling: ArchimedeanTiling,
    theta: float = 0.0,
    *,
    return_full: Literal[True],
    boundary: BoundaryHandling | str = BoundaryHandling.DISCARD,
) -> tuple[Array, Array, Array]: ...
@overload
def tile_into_rectangle(
    rect: Rectangle,
    tiling: ArchimedeanTiling,
    theta: float = 0.0,
    *,
    return_full: Literal[False] = False,
    boundary: BoundaryHandling | str = BoundaryHandling.DISCARD,
) -> tuple[Array, Array]: ...
def tile_into_rectangle(
    rect,
    tiling: ArchimedeanTiling,
    theta=0.0,
    *,
    return_full=False,
    boundary=BoundaryHandling.DISCARD,
):
    """Generate a tiling that fits into a rectangle of size (lx, ly). Does not offer the
    inclusion of periodic edges, or the option to add ghost nodes for periodicity
    handling. For this, use `tiling.tile_with_ghost_nodes` or
    `tiling.tile_with_periodic_edges` of a specific tiling directly.

    Args:
        rect: (x0, y0, lx, ly) defining the rectangle in cartesian coordinates.
        tiling: Tiling object to use.
        theta: Rotation angle of the tiling (in radians). Defaults to 0.0.
        return_full: If True, also returns all generated nodes before clipping. Defaults
            to False.
        boundary: Boundary handling strategy. One of 'cut', 'discard', 'cut_fill'. Defaults to
            'discard'.

    Returns:
        A tuple (nodes, edges) where nodes is an array of shape (N, 2)
        with the coordinates of the nodes inside the rectangle, edges is an array of
        shape (M, 2) with the indices of the nodes forming each edge
        If `return_full` is True, also returns nodes_all, an array of shape (K, 2) with
        the coordinates of all generated nodes before clipping.
    """
    x0, y0, lx, ly = rect
    x1 = x0 + lx
    y1 = y0 + ly
    cell_w = tiling.bravais_vectors()[0, 0]
    cell_h = tiling.bravais_vectors()[1, 1]
    boundary = BoundaryHandling(boundary)

    (p0, *_), width, height, _ = spanning_parallelogram(
        tiling.bravais_vectors().T, (x0, y0, lx, ly), -theta
    )
    ny = int(np.ceil(height / cell_h)) + 1
    nx = int(np.ceil(width / cell_w)) + 1

    nodes_all, edges_all = tiling.tile(nx, ny, x0=p0)
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    nodes_all = (R @ nodes_all.T).T

    # clipping
    def _get_inside_mask(nodes_all: Array) -> tuple[Array, Array]:
        inside = (
            (nodes_all[:, 0] >= x0 - EPS)
            & (nodes_all[:, 0] <= x1 + EPS)
            & (nodes_all[:, 1] >= y0 - EPS)
            & (nodes_all[:, 1] <= y1 + EPS)
        )
        keep = np.where(inside)[0]
        return inside, keep

    # filter edges
    inside, keep = _get_inside_mask(nodes_all)
    mask_edges = np.isin(edges_all, keep).all(axis=1)
    edges = edges_all[mask_edges]

    match boundary:
        case BoundaryHandling.DISCARD:
            pass
        case BoundaryHandling.CUT:
            add_points, add_edges = _add_boundary_cut_points(
                nodes_all, edges_all, inside, keep, lx, ly
            )
            if add_points.size > 0:
                nodes_all = np.vstack((nodes_all, add_points))
                edges = np.vstack((edges, add_edges))
        case BoundaryHandling.CUT_FILL:
            add_points, add_edges = _add_boundary_cut_points(
                nodes_all, edges_all, inside, keep, lx, ly
            )
            if add_points.size > 0:
                nodes_all = np.vstack((nodes_all, add_points))
                edges = np.vstack((edges, add_edges))

            inside, keep = _get_inside_mask(nodes_all)
            add_points, add_edges = _add_boundary_edges(
                nodes_all, inside, keep, (x0, y0, x1, y1)
            )
            nodes_all = np.vstack((nodes_all, add_points))
            edges = np.vstack((edges, add_edges))
            # if the original tiling had edges on the boundary, they are duplicated now
            # so we need to deduplicate
            # not very efficient but ok for now
            edges = _deduplicate_edges(edges)

    # renumbering
    inside, keep = _get_inside_mask(nodes_all)
    new_id = -np.ones(nodes_all.shape[0], dtype=int)
    new_id[keep] = np.arange(keep.size)
    nodes = nodes_all[keep]
    edges = new_id[edges]

    return (nodes, edges, nodes_all) if return_full else (nodes, edges)


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


def _add_boundary_cut_points(
    nodes_all: Array,
    edges_all: Array,
    inside: Array,
    keep: Array,
    lx: float,
    ly: float,
):
    """Add boundary cut points for edges crossing the rectangle boundary.

    Args:
        nodes_all: (N, 2) array with all node coordinates.
        edges_all: (M, 2) array with all edges (node indices).
        inside: (N,) bool array indicating which nodes are inside the rectangle.
        keep: (K,) array with indices of nodes to keep (inside).
        lx: Rectangle length in x direction.
        ly: Rectangle length in y direction.
    """
    # filter edges that pass the boundary
    # edges where one node is in keep and the other is not
    mask_boundary_edges = np.isin(edges_all, keep).any(axis=1) & ~np.isin(
        edges_all, keep
    ).all(axis=1)
    edges_boundary = edges_all[mask_boundary_edges]

    # boundary == "snap"
    # compute intersections
    edge_points = nodes_all[edges_boundary]  # (M,2,2)
    inter = _segment_rect_intersections_vec(edge_points, lx, ly)  # (M,2,2) with NaNs
    new_points = inter[~np.isnan(inter).any(axis=2)]  # (K,2)

    # points inside rectangle
    edge_boundary_inside = inside[edges_boundary]  # (M,2) bool
    # new segments
    new_points_idx = np.arange(inter.shape[0]) + nodes_all.shape[0]  # (M,)
    new_edges = np.vstack((edges_boundary[edge_boundary_inside], new_points_idx)).T

    return new_points, new_edges


def _segment_rect_intersections_vec(edge_points: Array, lx: float, ly: float) -> Array:
    """
    Vectorized Liang–Barsky clipping for many segments.

    Args:
        edge_points: (N, 2, 2) array with edge_points[i,0] = p_i, edge_points[i,1] = q_i
        lx: Rectangle length in x direction.
        ly: Rectangle length in y direction.

    Returns:
        (N, 2, 2) array intersections[i, k] = kth intersection point or NaN if absent.
    """
    p, q = edge_points[:, 0], edge_points[:, 1]  # (N,2)
    v = q - p  # (N,2)

    # Preallocate output: max 2 intersections per segment
    out = np.full((p.shape[0], 2, 2), np.nan, dtype=float)

    N = p.shape[0]

    # Liang–Barsky parameters for each boundary
    # For each segment, we gather:
    #   p_i = [-dx, dx, -dy, dy]
    #   q_i = [ p.x, lx-p.x, p.y, ly-p.y ]
    dx = v[:, 0]
    dy = v[:, 1]

    p_i = np.stack([-dx, dx, -dy, dy], axis=1)  # (N,4)
    q_i = np.stack([p[:, 0], lx - p[:, 0], p[:, 1], ly - p[:, 1]], axis=1)  # (N,4)

    # Initialize t0,t1
    t0 = np.zeros(N)
    t1 = np.ones(N)

    # Mask: which p_i == 0
    zero_mask = p_i == 0

    # Case p_i == 0 and q_i < 0 → reject
    reject_mask = np.any(zero_mask & (q_i < 0), axis=1)

    # For non-zero p_i compute t = q_i / p_i
    t = np.zeros_like(p_i, dtype=float)
    nz = ~zero_mask
    t[nz] = q_i[nz] / p_i[nz]

    # p_i < 0 → lower bound (t0)
    neg = p_i < 0
    # For entries where (p_i<0 & nz), update t0 = max(t0, t)
    t0 = np.maximum(t0, np.max(np.where(neg & nz, t, 0), axis=1))

    # p_i > 0 → upper bound (t1)
    pos = p_i > 0
    # For entries where (p_i>0 & nz), update t1 = min(t1, t)
    # Need to mask out irrelevant entries by replacing them with +inf
    t_pos = np.where(pos & nz, t, np.inf)
    t1 = np.minimum(t1, np.min(t_pos, axis=1))

    # Reject if t0 > t1
    reject_mask |= t0 > t1

    # Compute intersection points
    v = q - p
    # First intersection at t0 if 0 < t0 < 1
    cond0 = (t0 > 0) & (t0 < 1) & (~reject_mask)
    out[cond0, 0] = p[cond0] + t0[cond0, None] * v[cond0]

    # Second intersection at t1 if 0 < t1 < 1 and t1 != t0
    cond1 = (t1 > 0) & (t1 < 1) & (t1 != t0) & (~reject_mask)
    out[cond1, 1] = p[cond1] + t1[cond1, None] * v[cond1]

    return out


def _deduplicate_edges(edges: Array) -> Array:
    """Remove duplicate edges (regardless of orientation).

    Args:
        edges: (M, 2) array with edges (node indices).

    Returns:
        (K, 2) array with unique edges.
    """
    e = np.sort(edges, axis=1)
    e_view = e.view([("u", e.dtype), ("v", e.dtype)])
    uniq = np.unique(e_view)
    return uniq.view(e.dtype).reshape(-1, 2)


def _add_boundary_edges(
    nodes_all: Array,
    inside: Array,
    keep: Array,
    rect: tuple[float, float, float, float],
) -> tuple[Array, Array]:
    # boundary points
    x0, y0, x1, y1 = rect
    nodes_in = nodes_all[inside]

    n_b = np.isclose(nodes_in[:, 1], y0)
    n_b_id = keep[n_b]
    n_b_id = n_b_id[np.argsort(nodes_all[n_b_id, 0])]

    n_r = np.isclose(nodes_in[:, 0], x1)
    n_r_id = keep[n_r]
    n_r_id = n_r_id[np.argsort(nodes_all[n_r_id, 1])]

    n_t = np.isclose(nodes_in[:, 1], y1)
    n_t_id = keep[n_t]
    n_t_id = n_t_id[np.argsort(-nodes_all[n_t_id, 0])]

    n_l = np.isclose(nodes_in[:, 0], x0)
    n_l_id = keep[n_l]
    n_l_id = n_l_id[np.argsort(-nodes_all[n_l_id, 1])]

    # corner points (may be duplicated)
    corners = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    # find corner if exists
    corner_ids = []
    for i, cor in enumerate(corners):
        d = np.linalg.norm(nodes_in - cor, axis=1, ord=np.inf)
        c_id = np.where(d < EPS)[0]
        if c_id.size > 0:
            corner_ids.append(keep[c_id[0]])
        else:
            corner_ids.append(None)

    corner_points = []
    for i, (c_id, coords) in enumerate(zip(corner_ids, corners)):
        if c_id is None:
            # create corner point
            corner_points.append(coords)
            corner_ids[i] = nodes_all.shape[0] + len(corner_points) - 1
    corner_points = np.array(corner_points) if corner_points else np.empty((0, 2))

    # boundary segments
    corner_ids = np.array(corner_ids)

    edge_cycle = []
    for i, b_edge in enumerate([n_b_id, n_r_id, n_t_id, n_l_id]):
        b = b_edge
        if ~np.isin(corner_ids[i], b_edge):
            # already included
            b = np.insert(b, 0, corner_ids[i])
        if ~np.isin(corner_ids[(i + 1) % 4], b_edge):
            b = np.append(b, corner_ids[(i + 1) % 4])
        edge_cycle.append(b)

    # create edges
    edges_boundary = [
        np.vstack((b_edge[:-1], b_edge[1:])).T for b_edge in edge_cycle
    ]  # connect in order
    edges_boundary = (
        np.vstack(edges_boundary) if edges_boundary else np.empty((0, 2), dtype=int)
    )

    return corner_points, edges_boundary
