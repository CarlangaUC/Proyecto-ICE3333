"""Utilities used by ``recursivenodes``.

.. testsetup:: *

    from recursivenodes.utils import *
"""

import numpy as np
from math import comb


def multiindex_equal(d, k):
    """A generator for :math:`d`-tuple multi-indices whose sum is :math:`k`.

    Args:
        d (int): The length of the tuples
        k (int): The sum of the entries in the tuples

    Yields:
        tuple: tuples of length `d` whose entries sum to `k`, in lexicographic
        order.

    Example:
        >>> for i in multiindex_equal(3, 2): print(i)
        (0, 0, 2)
        (0, 1, 1)
        (0, 2, 0)
        (1, 0, 1)
        (1, 1, 0)
        (2, 0, 0)
    """
    if d <= 0:
        return
    if k < 0:
        return
    for i in range(k):
        for a in multiindex_equal(d-1, k-i):
            yield (i,) + a
    yield (k,) + (0,)*(d-1)


def multiindex_up_to(d, k):
    """A generator for :math:`d`-tuple multi-indices whose sum is at most
    :math:`k`.

    Args:
        d (int): The length of the tuples
        k (int): The maximum sum of the entries in the tuples

    Yields:
        tuple: tuples of length `d` whose entries sum to `k`, in lexicographic
        order.

    Example:
        >>> for i in multiindex_up_to(3, 2): print(i)
        (0, 0, 0)
        (0, 0, 1)
        (0, 0, 2)
        (0, 1, 0)
        (0, 1, 1)
        (0, 2, 0)
        (1, 0, 0)
        (1, 0, 1)
        (1, 1, 0)
        (2, 0, 0)
    """
    for a in multiindex_equal(d+1, k):
        yield a[0:d]


def multiindex_to_lex(alpha):
    """Get the lexicographic index of :math:`\\boldsymbol{\\alpha}` among all
    multiindices with the same length and sum.

    Args:
        alpha (tuple): A tuple of non-negative indices

    Returns:
        int: The index of :math:`\\boldsymbol{\\alpha}` in a lexicographically
        sorted list of multiindices with the same length and sum.

    Example:
        >>> multiindex_to_lex((1, 0, 1))
        3
    """
    d = len(alpha)
    if d <= 1:
        return 0
    k = alpha[0]
    n = sum(alpha)
    # We want the number of multiindices where the sum over the tail of alpha
    # is greater than n-k but at most n, which is the sum over all multiindices
    # of lenth d-1 whose sum is at most n, minus those whose sum is at most
    # (n-k)
    prev = comb(n + (d-1), (d-1)) - comb((n-k) + (d-1), (d-1))
    # Now we want the index of the tail in the list of all multiindices whose
    # first index is k
    kloc = multiindex_to_lex(alpha[1:])
    return prev + kloc


def npolys(d, k):
    """The number of polynomials up to a degree :math:`k` in :math:`d` dimensions.

    Args:
        d (int): The dimension of the space.
        k (int): The polynomial degree.

    Returns:
        int: :math:`\\binom{k+d}{d}`.

    Example:
        >>> npolys(3, 2)
        10
    """
    return comb(k+d, d)


def _equilateral_to_unit(d, x):
    if d > 1:
        # scale the vertical dimension
        x[..., d-1] /= ((d+1.)/(2*d))**0.5
        # make the projection onto the lesser dimensions right-angled
        _equilateral_to_unit(d-1, x[..., 0:d-1])
        # move the top vertex over first vertex
        x[..., 0:d-1] -= x[..., d-1, np.newaxis] / d


def _to_unit(x, domain):
    if domain == 'unit':
        return x.copy()
    if domain == 'biunit':
        return (x + 1) / 2
    if domain == 'barycentric':
        return x[..., 0:-1].copy()
    if domain == 'equilateral':
        d = x.shape[-1]
        z = x.copy()
        # scale edge length to 1
        z /= 2
        _equilateral_to_unit(d, z)
        z += 1/(d+1)  # shift the first vertex to zero
        return z
    else:
        raise NotImplementedError


def _unit_to_equilateral(d, x):
    if d > 1:
        # move the top vertex over the centroid
        x[..., 0:d-1] += x[..., d-1, np.newaxis] / d
        # make the projection onto the lesser dimensions equilateral
        _unit_to_equilateral(d-1, x[..., 0:d-1])
        # scale the vertical dimension
        x[..., d-1] *= ((d+1.)/(2*d))**0.5


def _from_unit(x, domain):
    if domain == 'unit':
        return x.copy()
    if domain == 'biunit':
        return x*2 - 1
    if domain == 'barycentric':
        z = 1 - x.sum(axis=-1)
        return np.concatenate((x, z[..., np.newaxis]), axis=-1)
    if domain == 'equilateral':
        d = x.shape[-1]
        z = x.copy()
        z -= 1/(d+1)  # shift centroid to zero
        _unit_to_equilateral(d, z)
        # scale edge length to 2
        z *= 2
        return z
    else:
        raise NotImplementedError


def coord_map(x, origin, image):
    '''Map coordinates from one reference `d`-simplex to another.

    Args:
        x (ndarray): 2D array of coordinates of shape ``(num_points,
            coordinate_dimension)``
        origin (str): Current domain of `x`, one of ``['unit', 'biunit',
            'equilateral', 'barycentric']``.
        image (str): Desired domain of mapped coordinates.

    Returns:
        ndarray: 2D array of coordinates in the new domain, shape
        ``(num_points, new_coordinated_dimension)``.

    See Also:
        ":ref:`domains`" for formal definitions of the domains.
    '''
    return _from_unit(_to_unit(x, domain=origin), domain=image)
