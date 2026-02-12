"""This module implements the 3 regular and 8 semiregular (Archimedean) tilings of the
plane.

Naming of tilings taken from
https://www.researchgate.net/publication/231156226_Archimedean_lattices_in_the_bound_states_of_wave_interacting_particles

General info about Archimedean tilings:
https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons#Archimedean.2C_uniform_or_semiregular_tilings
"""

from abc import abstractmethod
from typing import Sequence, TypeAlias

import numpy as np

Array: TypeAlias = np.ndarray


all_tilings: dict[str, type["ArchimedeanTiling"]] = {}


class _CollectAllTilingsMeta(type):
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)

        # include module name in name to avoid name conflicts with other tiling types
        # (e.g. orthogonal)
        mod = new_class.__module__.split(".")[-1]
        # if mod is __main__, use the class name only
        if mod != "__main__":
            name = f"{mod}.{name}"

        if name != "ArchimedeanTiling":
            all_tilings[name] = new_class
        return new_class


class ArchimedeanTiling(metaclass=_CollectAllTilingsMeta):
    def __init__(self, a: float = 1.0) -> None:
        self.a = a

    @abstractmethod
    def bravais_vectors(self) -> Array: ...
    @abstractmethod
    def tiling_basis(self) -> Array: ...
    @abstractmethod
    def edge_spec(self) -> Array:
        """Return (ne, 4) array of (i, j, di, dj)."""

    def tile(self, nx: int, ny: int, *, x0: Sequence = (0, 0)) -> tuple[Array, Array]:
        """Generate tiling in local coordinates.

        Args:
            nx: Number of tiles in x direction.
            ny: Number of tiles in y direction.
            x0: Origin offset. Defaults to (0, 0).

        Returns:
            A tuple (nodes, edges) where nodes is an array of shape (N, 2) with the
            coordinates of the nodes, and edges is an array of shape (M, 2) with the
            indices of the nodes forming each edge.
        """
        nodes = self._get_nodes(nx, ny, x0)
        edges = self._connectivity_from_spec(nx, ny, self.edge_spec())
        return nodes, edges

    def tile_with_periodic_edges(
        self, nx: int, ny: int, *, x0: Sequence = (0, 0)
    ) -> tuple[Array, Array]:
        """Generate tiling and wrap connectivity across right/top boundaries.

        No ghost nodes are created. Edges that would cross the right/top boundary are
        wrapped to the opposite side by using modulo arithmetic on node indices.

        Args:
            nx: Number of tiles in x direction.
            ny: Number of tiles in y direction.
            x0: Origin offset. Defaults to (0, 0).

        Returns:
            A tuple (nodes, edges) where nodes is an array of shape (N, 2) with the
            coordinates of the nodes, and edges is an array of shape (M, 2) with the
            indices of the nodes forming each edge. Edges that would cross the right/top
            boundary are wrapped to the opposite side.
        """
        nodes = self._get_nodes(nx, ny, x0)
        edges = self._connectivity_from_spec_periodic(nx, ny, self.edge_spec())
        return nodes, edges

    def tile_with_ghost_nodes(
        self, nx: int, ny: int, *, x0: Sequence = (0, 0)
    ) -> tuple[Array, Array, Array]:
        """Generate tiling and append periodic ghost nodes on right/top boundaries.

        Ghost nodes are added only when required by `edge_spec` entries that cross the
        right and/or top boundary, and represent wrapped left/bottom nodes translated by
        lattice periods.

        Returns:
            A tuple (nodes, edges, ghost_map), where:
            - nodes: (N + Ng, 2) coordinates including ghost nodes.
            - edges: (M, 2) connectivity using original and ghost indices.
            - ghost_map: (Ng, 2) array of (ghost_idx, real_idx), mapping each ghost
              node back to its wrapped in-domain node.
        """
        nodes = self._get_nodes(nx, ny, x0)

        edges, ghost_nodes, ghost_map = self._connectivity_from_spec_with_ghosts(
            nx, ny, self.edge_spec(), nodes
        )
        if ghost_nodes.size == 0:
            return nodes, edges, ghost_map
        return np.vstack((nodes, ghost_nodes)), edges, ghost_map

    def _get_nodes(self, nx: int, ny: int, x0: Sequence = (0, 0)) -> Array:
        IJ = np.stack(
            np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        nodes = IJ @ self.bravais_vectors() + np.array(x0)
        # map basis to all nodes
        nodes = (nodes[:, None, :] + self.tiling_basis()[None, :, :]).reshape(-1, 2)
        return nodes

    def _connectivity_from_spec(self, nx: int, ny: int, spec: Array) -> Array:
        # spec: (ne, 4) = (i, j, di, dj)
        nb = self.tiling_basis().shape[0]

        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # noqa: E741
        I = I.ravel()  # noqa: E741
        J = J.ravel()
        cell_id = I * ny + J  # (nc,)

        edges = []

        for isite, jsite, dI, dJ in spec:
            I2 = I + dI
            J2 = J + dJ
            mask = (I2 >= 0) & (I2 < nx) & (J2 >= 0) & (J2 < ny)
            if not np.any(mask):
                continue

            cell1 = cell_id[mask]
            cell2 = I2[mask] * ny + J2[mask]

            n1 = cell1 * nb + isite
            n2 = cell2 * nb + jsite
            edges.append(np.stack([n1, n2], axis=1))

        if not edges:
            return np.empty((0, 2), dtype=int)

        edges = np.vstack(edges)

        # TODO: decide if this should be kept for safety
        # sort & unique
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        return edges

    def _connectivity_from_spec_periodic(self, nx: int, ny: int, spec: Array) -> Array:
        # spec: (ne, 4) = (i, j, di, dj)
        nb = self.tiling_basis().shape[0]

        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # noqa: E741
        I = I.ravel()  # noqa: E741
        J = J.ravel()
        cell_id = I * ny + J  # (nc,)

        edges = []

        for isite, jsite, dI, dJ in spec:
            I2 = (I + dI) % nx
            J2 = (J + dJ) % ny

            n1 = cell_id * nb + isite
            n2 = (I2 * ny + J2) * nb + jsite
            edges.append(np.stack([n1, n2], axis=1))

        if not edges:
            return np.empty((0, 2), dtype=int)

        edges = np.vstack(edges)

        # sort & unique
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        return edges

    def _connectivity_from_spec_with_ghosts(
        self, nx: int, ny: int, spec: Array, nodes: Array
    ) -> tuple[Array, Array, Array]:
        # spec: (ne, 4) = (i, j, di, dj)
        nb = self.tiling_basis().shape[0]
        a1, a2 = self.bravais_vectors()

        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")  # noqa: E741
        I = I.ravel()  # noqa: E741
        J = J.ravel()
        cell_id = I * ny + J  # (nc,)

        edges = []
        ghost_nodes = []
        ghost_map = []
        ghost_lookup: dict[tuple[int, int, int], int] = {}
        base_n = nodes.shape[0]

        for isite, jsite, dI, dJ in spec:
            I2 = I + dI
            J2 = J + dJ

            in_bounds = (I2 >= 0) & (I2 < nx) & (J2 >= 0) & (J2 < ny)
            if np.any(in_bounds):
                cell1 = cell_id[in_bounds]
                cell2 = I2[in_bounds] * ny + J2[in_bounds]
                n1 = cell1 * nb + isite
                n2 = cell2 * nb + jsite
                edges.append(np.stack([n1, n2], axis=1))

            # Ghosts only for right/top crossings.
            cross_rt = ~in_bounds & ((I2 >= nx) | (J2 >= ny))
            if not np.any(cross_rt):
                continue

            src = cell_id[cross_rt] * nb + isite
            I2c = I2[cross_rt]
            J2c = J2[cross_rt]
            Iw = I2c % nx
            Jw = J2c % ny
            real = (Iw * ny + Jw) * nb + jsite
            sx = I2c // nx
            sy = J2c // ny

            for n1, nreal, shx, shy in zip(src, real, sx, sy, strict=False):
                key = (int(nreal), int(shx), int(shy))
                if key not in ghost_lookup:
                    gidx = base_n + len(ghost_nodes)
                    ghost_lookup[key] = gidx
                    ghost_nodes.append(nodes[nreal] + shx * nx * a1 + shy * ny * a2)
                    ghost_map.append([gidx, int(nreal)])
                edges.append(np.array([[int(n1), ghost_lookup[key]]], dtype=int))

        if edges:
            edges_arr = np.vstack(edges)
            edges_arr = np.sort(edges_arr, axis=1)
            edges_arr = np.unique(edges_arr, axis=0)
        else:
            edges_arr = np.empty((0, 2), dtype=int)

        if ghost_nodes:
            ghosts_arr = np.asarray(ghost_nodes, dtype=float)
            ghost_map_arr = np.asarray(ghost_map, dtype=int)
        else:
            ghosts_arr = np.empty((0, 2), dtype=float)
            ghost_map_arr = np.empty((0, 2), dtype=int)

        return edges_arr, ghosts_arr, ghost_map_arr


class Hex(ArchimedeanTiling):
    def bravais_vectors(self) -> np.ndarray:
        # 60 deg hex bravais lattice
        return np.array(
            [
                [self.a, 0.0],
                [0.5 * self.a, np.sqrt(3) / 2 * self.a],
            ]
        )

    def tiling_basis(self) -> np.ndarray:
        # two point basis -> hexagonal tiling
        return np.array(
            [
                [0.0, 0.0],
                [0.5 * self.a, np.sqrt(3) / 6 * self.a],
            ]
        )

    def edge_spec(self) -> np.ndarray:
        # (i, j, di, dj)
        return np.array(
            [
                [0, 1, 0, 0],
                [1, 0, +1, 0],  # right neighbor cell
                [1, 0, 0, +1],  # top neighbor
            ],
            dtype=int,
        )


class Triangular(ArchimedeanTiling):
    def bravais_vectors(self) -> np.ndarray:
        # 60 deg triangular bravais lattice
        return np.array(
            [
                [self.a, 0.0],
                [0.5 * self.a, np.sqrt(3) / 2 * self.a],
            ]
        )

    def tiling_basis(self) -> np.ndarray:
        # single point basis -> triangular tiling
        return np.array([[0.0, 0.0]])

    def edge_spec(self) -> Array:
        # (i, j, di, dj)
        return np.array(
            [
                [0, 0, +1, 0],  # right neighbor cell
                [0, 0, 0, +1],  # top neighbor
                [0, 0, +1, -1],  # down-right neighbor
            ],
            dtype=int,
        )


class Kagome(ArchimedeanTiling):
    def bravais_vectors(self) -> Array:
        return np.array(
            [
                [self.a, 0.0],
                [0.5 * self.a, np.sqrt(3) / 2 * self.a],
            ]
        )

    def tiling_basis(self) -> Array:
        return np.array(
            [
                [0.0, 0.0],
                [0.5 * self.a, 0.0],
                [0.25 * self.a, np.sqrt(3) / 4 * self.a],
            ]
        )

    def edge_spec(self) -> Array:
        # (i, j, di, dj)
        return np.array(
            [
                [0, 1, 0, 0],
                [0, 2, 0, 0],
                [1, 2, 0, 0],
                [1, 0, +1, 0],  # right neighbor cell
                [2, 0, 0, +1],  # top neighbor
                [1, 2, +1, -1],  # down-right neighbor
            ],
            dtype=int,
        )


class Square(ArchimedeanTiling):
    def bravais_vectors(self):
        return np.array([[self.a, 0.0], [0.0, self.a]])

    def tiling_basis(self):
        return np.array([[0.0, 0.0]])

    def edge_spec(self):
        # right, up
        return np.array(
            [
                [0, 0, +1, 0],
                [0, 0, 0, +1],
            ],
            int,
        )


class TruncatedHex(ArchimedeanTiling):
    # 3-12-12 tiling
    def bravais_vectors(self):
        a = self.a
        return np.array(
            [
                [a, 0],
                [0.5 * a, np.sqrt(3) / 2 * a],
            ]
        )

    def tiling_basis(self):
        l = 1 / (2 + np.sqrt(3)) * self.a  # noqa: E741
        return np.array(
            [
                [0.0, 0.0],
                [l, 0],
                [l / 2, np.sqrt(3) / 2 * l],
                [l / 2, (1 + np.sqrt(3) / 2) * l],
                [l, (1 + np.sqrt(3)) * l],
                [0, (1 + np.sqrt(3)) * l],
            ]
        )

    def edge_spec(self):
        return np.array(
            [
                # first, internal cell connections
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [2, 0, 0, 0],
                [2, 3, 0, 0],
                [3, 4, 0, 0],
                [4, 5, 0, 0],
                [5, 3, 0, 0],
                # connect to neighbor cells
                [5, 1, -1, +1],
                [4, 0, 0, +1],
            ],
            dtype=int,
        )


class GreatRhombiTriHex(ArchimedeanTiling):
    # 4-6-12 tiling
    def bravais_vectors(self):
        a = self.a
        return np.array(
            [
                [a, 0],
                [0.5 * a, np.sqrt(3) / 2 * a],
            ]
        )

    def tiling_basis(self):
        l = 1 / (3 + np.sqrt(3)) * self.a  # noqa: E741
        hex_1 = np.array(
            [
                [0, 0],
                [l, 0],
                [1.5 * l, np.sqrt(3) / 2 * l],
                [l, np.sqrt(3) * l],
                [0, np.sqrt(3) * l],
                [-0.5 * l, np.sqrt(3) / 2 * l],
            ]
        )
        hex_2 = hex_1 + np.array([self.a / 2, (np.sqrt(3) / 2 + 0.5) * l])
        return np.vstack([hex_1, hex_2])

    def edge_spec(self):
        hex1 = np.array(
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [2, 3, 0, 0],
                [3, 4, 0, 0],
                [4, 5, 0, 0],
                [5, 0, 0, 0],
            ]
        )
        hex2 = hex1 + np.array([6, 6, 0, 0])
        conn_hex = np.array(
            [
                [2, 6, 0, 0],
                [3, 11, 0, 0],
                [7, 5, +1, 0],
                [8, 4, +1, 0],
                [9, 1, 0, +1],
                [10, 0, 0, +1],
            ]
        )
        return np.vstack([hex1, hex2, conn_hex], dtype=int)


class TruncatedSquare(ArchimedeanTiling):
    # 4-8-8 tiling
    def bravais_vectors(self):
        return np.array([[self.a, 0.0], [0.0, self.a]])

    def tiling_basis(self):
        l = 1 / (1 + np.sqrt(2)) * self.a  # noqa: E741
        l_diag = np.sqrt(2) / 2 * l
        return np.array([[-l_diag, 0], [0, -l_diag], [l_diag, 0], [0, l_diag]])

    def edge_spec(self):
        return np.array(
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [2, 3, 0, 0],
                [3, 0, 0, 0],
                # connect to neighbor cells
                [2, 0, +1, 0],
                [3, 1, 0, +1],
            ],
            int,
        )


class SmallRhombiTriHex(ArchimedeanTiling):
    # 3-4-6-4 tiling
    def bravais_vectors(self):
        a = self.a
        return np.array(
            [
                [a, 0],
                [0.5 * a, np.sqrt(3) / 2 * a],
            ]
        )

    def tiling_basis(self):
        l = 1 / (1 + np.sqrt(3)) * self.a  # noqa: E741
        return np.array(
            [
                [0.0, 0.0],
                [l, 0],
                [l, l],
                [0, l],
                [(1 + np.sqrt(3) / 2) * l, 3 / 2 * l],
                [l / 2, (1 + np.sqrt(3) / 2) * l],
            ]
        )

    def edge_spec(self):
        return np.array(
            [
                # first, internal cell connections
                [0, 1, 0, 0],
                [0, 3, 0, 0],
                [1, 2, 0, 0],
                [2, 3, 0, 0],
                [4, 2, 0, 0],
                [3, 5, 0, 0],
                [2, 5, 0, 0],
                # connect to neighbor cells
                [1, 5, +1, -1],
                [4, 3, +1, 0],
                [5, 0, 0, +1],
                [4, 0, 0, +1],
                [4, 1, 0, +1],
            ],
            dtype=int,
        )


class SnubHex(ArchimedeanTiling):
    # 3-3-3-3-6 tiling
    def bravais_vectors(self):
        a = self.a
        s_t, c_t = np.sqrt(3) / np.sqrt(7), 2 / np.sqrt(7)
        s_b, c_b = -np.sqrt(3) / 2 / np.sqrt(7), 5 / 2 / np.sqrt(7)
        return a * np.array(
            [
                [c_b, s_b],
                [c_t, s_t],
            ]
        )

    def tiling_basis(self):
        a = self.a
        l = 1 / np.sqrt(7) * a  # noqa: E741
        h = np.sqrt(3) / 2 * l
        return np.array(
            [
                [0, 0],
                [l, 0],
                [2 * l, 0],
                [0.5 * l, h],
                [1.5 * l, h],
                [2.5 * l, h],
            ]
        )

    def edge_spec(self):
        return np.array(
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [4, 5, 0, 0],
                [0, 3, 0, 0],
                [3, 1, 0, 0],
                [1, 4, 0, 0],
                [4, 2, 0, 0],
                [2, 5, 0, 0],
                # connect to neighbor cells
                [2, 3, +1, 0],
                [2, 0, +1, 0],
                [4, 0, 0, +1],
                [5, 0, 0, +1],
                [5, 1, 0, +1],
                [5, 3, +1, 0],
            ]
        )


class SnubSquare(ArchimedeanTiling):
    # 3-3-4-3-4 tiling
    def bravais_vectors(self):
        a_diag = self.a * np.sqrt(2) / 2
        return np.array([[a_diag, -a_diag], [a_diag, a_diag]])

    def tiling_basis(self):
        l = 1 / 2 / np.cos(np.pi / 12) * self.a  # noqa: E741
        h = np.sqrt(3) / 2 * l
        return np.array([[0, 0], [h, -l / 2], [h, l / 2], [2 * h, 0]])

    def edge_spec(self):
        return np.array(
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [2, 3, 0, 0],
                [1, 3, 0, 0],
                [0, 2, 0, 0],
                # connect to neighbor cells
                [1, 0, +1, 0],
                [2, 0, 0, +1],
                [3, 2, +1, 0],
                [3, 0, +1, +1],
                [3, 1, 0, +1],
            ],
            dtype=int,
        )


class ElongatedTriangular(ArchimedeanTiling):
    # 3-3-3-4-4 tiling
    def bravais_vectors(self):
        a = self.a
        return np.array(
            [
                [a, 0],
                [a / 2, a * (1 + np.sqrt(3) / 2)],
            ]
        )

    def tiling_basis(self):
        l = self.a  # noqa: E741
        return np.array(
            [
                [0.0, 0.0],
                [l, 0.0],
                [l / 2, np.sqrt(3) / 2 * l],
            ]
        )

    def edge_spec(self):
        return np.array(
            [
                # first, internal cell connections
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [2, 0, 0, 0],
                # connect to neighbor cells
                [2, 0, 0, +1],
                [2, 2, +1, 0],
            ],
            dtype=int,
        )
