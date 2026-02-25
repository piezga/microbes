import numpy as np


def flat2triumat_dim(k):
    """Return the dimension of the matrix hosting ``k`` triangular elements.

    :param k: the number of elements in the upper triangular
        part of the corresponding square matrix, excluding the diagonal
    :type k: int
    :returns: dimension of the corresponding square matrix
    :rtype: int
    """
    return int(0.5 + np.sqrt(0.25 + 2 * k))

def flat2triumat_idx(k, n):
    """Convert an array index into the index couple of a triangular matrix.

    ``k`` is the index of an array of length :math:`\\binom{n}{2}{2}`,
    which contains the elements of an upper triangular matrix of dimension
    ``n`` excluding the diagonal. The function returns the index couple
    :math:`(i, j)` that corresponds to the entry ``k`` of the flat array.

    .. note::
        * :math:`k \\in \left[0,\\ldots, \\binom{n}{2} - 1\\right]`
        * returned indices:
            * :math:`i \\in [0,\\ldots, n - 1]`
            * :math:`j \\in [i + 1,\\ldots, n - 1]`

    :param k: flattened array index
    :type k: int
    :param n: dimension of the square matrix
    :type n: int
    :returns: matrix index tuple (row, column)
    :rtype: tuple
    """
    # row index of array index k in the the upper triangular part of the
    # square matrix
    r = n - 2 - int(0.5 * np.sqrt(-8 * k + 4 * n * (n - 1) - 7) - 0.5)
    # column index of array index k in the the upper triangular part of the
    # square matrix
    c = k + 1 + r * (3 - 2 * n + r) / 2
    return r, c

