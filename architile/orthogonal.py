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
