"""Contains periodic patterns with orthogonal periodicity."""

import numpy as np

from architile.tiling import ArchimedeanTiling
from architile.tiling import Square as Square  # re-exported for convenience


class Triangular(ArchimedeanTiling):
    """Same as `tiling.Triangular`, but with orthogonal periodicity."""

    def bravais_vectors(self):
        a = self.a
        return np.array([[a, 0], [0, np.sqrt(3) * a]])

    def tiling_basis(self):
        a = self.a
        return np.array(
            [
                [0, 0],
                [a / 2, np.sqrt(3) / 2 * a],
            ]
        )

    def edge_spec(self):
        return np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 1, 0],
            ]
        )


class Hex(ArchimedeanTiling):
    """Same as `tiling.Hex`, but with orthogonal periodicity.

    Bravais vectors: (3a, 0), (0, sqrt(3)a)

    Unit cell pattern:
            1 ____________2
             /            \
            /              \
           /                \
         0/                  \3
    """

    def bravais_vectors(self):
        a = self.a
        return np.array([[3 * a, 0], [0, np.sqrt(3) * a]])

    def tiling_basis(self):
        a = self.a
        return np.array(
            [
                [0, 0],
                [a / 2, np.sqrt(3) / 2 * a],
                [3 * a / 2, np.sqrt(3) / 2 * a],
                [2 * a, 0],
            ]
        )

    def edge_spec(self):
        return np.array(
            [
                [0, 1, 0, 0],
                [1, 2, 0, 0],
                [2, 3, 0, 0],
                [3, 0, +1, 0],
                [1, 0, 0, +1],
                [2, 3, 0, +1],
            ],
            dtype=int,
        )

