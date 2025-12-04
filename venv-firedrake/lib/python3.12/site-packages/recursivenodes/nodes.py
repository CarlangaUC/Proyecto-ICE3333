"""Node sets for the `d`-simplex implemented by ``recursivenodes``"""

import numpy as np

from recursivenodes.utils import (npolys,
                                  multiindex_equal,
                                  multiindex_up_to,
                                  coord_map)
from recursivenodes.quadrature import gaussjacobi, lobattogaussjacobi
from recursivenodes.polynomials import proriolkoornwinderdubinervandermonde


def equispaced(d, n, domain='biunit'):
    '''Equispaced nodes for polynomials up to degree `n` on the `d`-simplex.

    Args:
        d (int): The dimension of the simplex.
        n (int): The polynomial degree
        domain (str, optional) -- The domain of the simplex.  See
            ":ref:`domains`" for the choices and their formal definitions.

    Returns:
        ndarray: Equispaced nodes as a 2D array with one row for each of
        `\\binom{n+d}{d}` nodes, and `d` columns for coordinates (or `d+1` if
        ``domain='barycentric'``).

    Example:

        .. plot::
           :include-source: True

           >>> import matplotlib.pyplot as plt
           >>> from numpy import eye
           >>> from recursivenodes.nodes import equispaced
           >>> from recursivenodes.utils import coord_map
           >>> nodes = equispaced(2, 7, domain='equilateral')
           >>> corners = coord_map(eye(3), 'barycentric', 'equilateral')
           >>> plt.plot(corners[[0,1,2,0],0], corners[[0,1,2,0],1])
           [<matplotlib.lines.Line2D object at ...>]
           >>> plt.scatter(nodes[:,0], nodes[:,1])
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.gca().set_aspect('equal')
           >>> plt.title('Equispaced Nodes')
           Text(0.5, 1.0, 'Equispaced Nodes')
           >>> plt.show()
    '''
    N = npolys(d, n)
    x = np.ndarray((N, d))
    for (k, i) in enumerate(multiindex_up_to(d, n)):
        x[k, :] = np.array(i) / n
    return coord_map(x, 'unit', domain)


def equispaced_interior(d, n, domain='biunit'):
    '''Equispaced nodes for polynomials up to degree `n` on the `d`-simplex,
    all of which are in the interior.

    Args:
        d (int): The dimension of the simplex.
        n (int): The polynomial degree
        domain (str, optional): The domain of the simplex.  See
            ":ref:`domains`" for the choices and their formal definitions.

    Returns:
        ndarray: Equispaced interior nodes as a 2D array with one row for each
        of `\\binom{n+d}{d}` nodes, and `d` columns for coordinates (or `d+1`
        if ``domain='barycentric'``).

    Example:

        .. plot::
           :include-source: True

           >>> import matplotlib.pyplot as plt
           >>> from numpy import eye
           >>> from recursivenodes.nodes import equispaced_interior
           >>> from recursivenodes.utils import coord_map
           >>> nodes = equispaced_interior(2, 7, domain='equilateral')
           >>> corners = coord_map(eye(3), 'barycentric', 'equilateral')
           >>> plt.plot(corners[[0,1,2,0],0], corners[[0,1,2,0],1])
           [<matplotlib.lines.Line2D object at ...>]
           >>> plt.scatter(nodes[:,0], nodes[:,1])
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.gca().set_aspect('equal')
           >>> plt.title('Equispaced Interior Nodes')
           Text(0.5, 1.0, 'Equispaced Interior Nodes')
           >>> plt.show()
    '''
    N = npolys(d, n)
    x = np.ndarray((N, d))
    for (k, i) in enumerate(multiindex_up_to(d, n)):
        x[k, :] = (np.array(i) + 1/(d+1)) / (n+1)
    return coord_map(x, 'unit', domain)


def blyth_luo_pozrikidis(d, n, x=None, domain='biunit'):
    '''Create Blyth-Luo-Pozrikidis nodes from a 1D node set for polynomials up
    to degree `n` on the `d`-simplex.

    Notes:

        The Blyth-Luo-Pozrikidis rule places an interior node with multi-index
        `\\boldsymbol{\\alpha}` at the barycentric point `\\boldsymbol{b}(\\boldsymbol{\\alpha})` such that

        .. math::
           :label: blp

           b_i(\\boldsymbol{\\alpha}) = x_{n,\\boldsymbol{\\alpha}_i} + \\frac{1}{d}(1 - \\sum_{j\\neq i}
           x_{n,\\boldsymbol{\\alpha}_j}).

        Points on the boundary look like `(d-1)`-simplex Blyth-Luo-Pozrikidis
        nodes.

    Args:
        d (int): The dimension of the simplex.
        n (int): The polynomial degree
        x (ndarray, optional): 1D node set on `[0, 1]` with `n+1` points.
            Lobatto-Gauss-Legendre nodes are used if  ``x=None``.
        domain (str, optional): The domain of the simplex.  See
            ":ref:`domains`" for the choices and their formal definitions.

    Returns:
        ndarray: Blyth-Luo-Pozrikidis nodes as a 2D array with one row for each
        of `\\binom{n+d}{d}` nodes, and `d` columns for coordinates (or `d+1`
        if ``domain='barycentric'``).

    Example:

        This plot shows the Blyth-Luo-Pozrikidis nodes with lines connecting
        the Lobatto-Gauss-Legendre nodes on the edges.  The definition in
        :eq:`blp` is designed to place the nodes in the centroids of the
        triangles created by the intersecting lines.

        .. plot::
           :include-source: True

           >>> import matplotlib.pyplot as plt
           >>> from numpy import eye
           >>> from recursivenodes.nodes import blyth_luo_pozrikidis
           >>> from recursivenodes.utils import coord_map, multiindex_equal
           >>> nodes = blyth_luo_pozrikidis(2, 7, domain='equilateral')
           >>> corners = coord_map(eye(3), 'barycentric', 'equilateral')
           >>> # plot grid lines
           >>> for (i, al) in enumerate(multiindex_equal(3, 7)):
           ...     if min(al) > 0 or max(al) == sum(al): continue
           ...     for (j, be) in enumerate(multiindex_equal(3, 7)):
           ...         if (j <= i or min(be) > 0 or max(be) != max(al)
           ...                 or sum(be) != sum(al)):
           ...             continue
           ...         for d in range(3):
           ...             if al[d] == be[d] and al[d] != 0:
           ...                 _ = plt.plot(nodes[[i,j],0], nodes[[i,j],1],
           ...                              linestyle='--', c='grey')
           >>> plt.plot(corners[[0,1,2,0],0], corners[[0,1,2,0],1])
           [<matplotlib.lines.Line2D object at ...>]
           >>> plt.scatter(nodes[:,0], nodes[:,1])
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.gca().set_aspect('equal')
           >>> plt.title('Blyth-Luo-Pozrikidis Nodes')
           Text(0.5, 1.0, 'Blyth-Luo-Pozrikidis Nodes')
           >>> plt.show()

    References:

        :cite:`BlPo06,LuPo06`
    '''
    if x is None:
        x = coord_map(lobattogaussjacobi(n+1)[0].reshape(n+1, 1), 'biunit', domain)
        if domain != 'barycentric':
            x = x[:, 0]
    n = len(x) - 1
    x = x.reshape(x.shape + (1,))
    y = coord_map(x, domain, 'unit')
    y = y[:, 0]
    N = npolys(d, n)
    xd = np.zeros((N, d))
    for (k, i) in enumerate(multiindex_equal(d+1, n)):
        numzero = sum([int(y[m] == 0.) for m in i])
        weight = 0.
        for j in range(d+1):
            ibutj = i[0:j] + i[(j+1):]
            yibutj = [y[m] for m in ibutj]
            thisnumzero = sum([int(z == 0) for z in yibutj])
            if thisnumzero < numzero:
                continue
            yj = 1. - sum(yibutj)
            weight += 1
            for m in range(j):
                if (m < d):
                    xd[k, m] += y[i[m]]
            if (j < d):
                xd[k, j] += yj
            for m in range(j+1, d+1):
                if (m < d):
                    xd[k, m] += y[i[m]]
        xd[k, :] /= weight
    return coord_map(xd, 'unit', domain)


def _warburton_b(d, xt, i):
    b = np.ones((xt.shape[0], 1))
    tol = 1.e-8
    bdry = np.zeros(b.shape)
    for j in range(d+1):
        if j == i:
            continue
        nonzero = np.abs(2. * xt[:, j] + xt[:, i]) > tol
        b[nonzero, 0] *= ((2. * xt[nonzero, j]) /
                          (2. * xt[nonzero, j] + xt[nonzero, i]))
        bdry[~nonzero, 0] += 1.
    isbdry = bdry[:, 0] > 0.
    if d == 2:
        '''The implementation of warp and blend, evident in the Lebesgue
        constants in the paper and in modepy
        (https://github.com/inducer/modepy/blob/master/modepy/nodes.py)
        uses the blending scaling (2 * l2) * (2 * l3) / (1. - (l2 - l3)**2)
        instead of (2 * l2) * (2 * l3) / ((2 * l2 + l1)*(2 * l3 + l1)).

        If l1 + l2 + l3 = 1, this is equivalent, because

        (1. - (l2 - l3)**2) = (1. + (l2 - l3))*(1. - (l2 - l3))
                            = (l1+l2+l3 +l2-l3)*(l1+l2+l3 -l2+l3)
                            = (2*l2+l1)*(2.*l3+l1)

        BUT, when we are blending for higher dimensions, we may be using a
        subset of barycentric coordinates, i.e. l1 + l2 + l3 + l4 = 1, l4 > 0,
        and we are computing a face warp for the l4 face, so we use only the
        (l1,l2,l3) coordinate.  In this case l1+l2+l3 != 1, so the two blending
        scalings are not equivalent.  I'm choosing to follow the implementation
        so that I can reproduce the computed Lebesgue constants.
        '''
        i2 = (i + 1) % 3
        i3 = (i + 2) % 3
        b[~isbdry, 0] = (4. * xt[~isbdry, i2] * xt[~isbdry, i3] /
                         (1. - (xt[~isbdry, i2]-xt[~isbdry, i3])**2))
    b[isbdry, 0] = 1. / (bdry[isbdry, 0] + 1.)
    return b


def _warburton_g(d, n, xt, alpha=0., x=None):
    g = np.zeros(xt.shape)
    if (d == 1):
        if x is None:
            x, _ = lobattogaussjacobi(n+1)
        xe = np.linspace(-1., 1., n+1, endpoint=True)
        diff = (x - xe) / 2
        V = proriolkoornwinderdubinervandermonde(1, n, xe.reshape((n+1, 1)))
        P = np.linalg.solve(V, diff)
        T = proriolkoornwinderdubinervandermonde(1, n, xt[:, [0]] - xt[:, [1]])
        g[:, 0] = T.dot(P)
        g[:, 1] = -g[:, 0]
    else:
        for i in range(d+1):
            rem = [j for j in range(d+1) if j != i]
            gi = _warburton_g(d-1, n, xt[:, rem], alpha, x)
            bi = _warburton_b(d, xt, i)
            g[:, rem] += (1. + (alpha * xt[:, [i]])**2) * gi * bi
    return g


#: A dictionary of optimal values of the blending parameter `\alpha` computed
#: in :cite:`Warb06`.  Keyed by ``(d, n)`` tuples.
#:
#: Example:
#:     >>> warburton_alpha[(2,7)]
#:     1.0999
warburton_alpha = {
    (2, 3):  1.4152, (3, 3):  0.0000,
    (2, 4):  0.1001, (3, 4):  0.1002,
    (2, 5):  0.2751, (3, 5):  1.1332,
    (2, 6):  0.9808, (3, 6):  1.5608,
    (2, 7):  1.0999, (3, 7):  1.3413,
    (2, 8):  1.2832, (3, 8):  1.2577,
    (2, 9):  1.3648, (3, 9):  1.1603,
    (2, 10): 1.4773, (3, 10): 1.0153,
    (2, 11): 1.4959, (3, 11): 0.6080,
    (2, 12): 1.5743, (3, 12): 0.4523,
    (2, 13): 1.5770, (3, 13): 0.8856,
    (2, 14): 1.6223, (3, 14): 0.8717,
    (2, 15): 1.6258, (3, 15): 0.9655,
}


def warburton(d, n, x=None, alpha=None, domain='biunit'):
    '''Warburton *warp & blend* nodes from a 1D node set for polynomials up
    to degree `n` on the `d`-simplex.

    Notes:

        The Warburton *warp & blend* nodes define a node's coordinates as a
        displacement from equispaced coordinates by blending together
        distortion maps of the `d`-simplex that warp the edge nodes
        to match the 1D node set.  One optimization parameter `\\boldsymbol{\\alpha}`
        controls the blending in the interior of the simplex.

    Args:
        d (int): The dimension of the simplex.
        n (int): The polynomial degree
        x (ndarray, optional): 1D node set on `[0, 1]` with `n+1` points.
            Lobatto-Gauss-Legendre nodes are used if ``x=None``.
        alpha (float, optional): The blending parameter. If ``alpha=None``, a
            precomputed optimal parameter is used if known.
        domain (str, optional): The domain of the simplex.  See
            ":ref:`domains`" for the choices and their formal definitions.

    Returns:
        ndarray: Warburton *warp & blend* nodes as a 2D array with one row for
        each of `\\binom{n+d}{d}` nodes, and `d` columns for coordinates (or
        `d+1` if ``domain='barycentric'``).

    Example:

        .. plot::
           :include-source: True

           >>> import matplotlib.pyplot as plt
           >>> from numpy import eye
           >>> from recursivenodes.nodes import warburton
           >>> from recursivenodes.utils import coord_map
           >>> nodes = warburton(2, 7, domain='equilateral')
           >>> corners = coord_map(eye(3), 'barycentric', 'equilateral')
           >>> plt.plot(corners[[0,1,2,0],0], corners[[0,1,2,0],1])
           [<matplotlib.lines.Line2D object at ...>]
           >>> plt.scatter(nodes[:,0], nodes[:,1])
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.gca().set_aspect('equal')
           >>> plt.title('Warburton Warp & Blend Nodes')
           Text(0.5, 1.0, 'Warburton Warp & Blend Nodes')
           >>> plt.show()

    References:

        :cite:`Warb06`
    '''
    if not (x is None):
        x = coord_map(x, domain, 'biunit')
    if alpha is None:
        try:
            alpha = warburton_alpha[(d, n)]
        except KeyError:
            alpha = 0.
    xt = equispaced(d, n, domain='barycentric')
    g = _warburton_g(d, n, xt, alpha, x)
    xt += g
    return coord_map(xt, 'barycentric', domain)


class NodeFamily:
    '''Family of nodes on the unit interval.  This class essentially is a
    lazy-evaluate-and-cache dictionary: the user passes a routine to evaluate
    entries for unknown keys'''

    def __init__(self, f):
        self._f = f
        self._cache = {}

    def __getitem__(self, key):
        try:
            return self._cache[key]
        except KeyError:
            value = self._f(key)
            self._cache[key] = value
            return value


# For each family, family[n] should be a a set of n+1 points in [0,1] in
# increasing order that is symmetric about 1/2, family[n] should be None if the
# family does not have a representative for that index
#
# We predefine:
#
#   - shifted Lobatto-Gauss-Legendre  (lgl_family)
#   - shifted Lobatto-Gauss-Chebyshev (lgc_family)
#   - shifted Gauss-Legendre          (gl_family)
#   - shifted Gauss-Chebyshev         (gc_family)
#   - equispaced, including endpoints (equi_family)
#   - equispaced, interior            (equi_interior_family)
lgl_family = NodeFamily(lambda n:
                        coord_map(lobattogaussjacobi(n+1)[0], 'biunit', 'unit')
                        if n > 0 else np.array([0.5]))
lgc_family = NodeFamily(lambda n:
                        coord_map(lobattogaussjacobi(n+1, -0.5, -0.5)[0],
                                  'biunit', 'unit')
                        if n > 0 else np.array([0.5]))
gl_family = NodeFamily(lambda n:
                       coord_map(gaussjacobi(n+1)[0], 'biunit', 'unit'))
gc_family = NodeFamily(lambda n:
                       coord_map(gaussjacobi(n+1, -0.5, -0.5)[0],
                                 'biunit', 'unit'))
equi_family = NodeFamily(lambda n:
                         np.linspace(0., 1., n+1) if n > 0 else
                         np.array([0.5]))
equi_interior_family = NodeFamily(lambda n:
                                  np.linspace(0.5/(n+1), 1.-0.5/(n+1), n+1))


def _recursive(d, n, alpha, family):
    '''The barycentric d-simplex coordinates for a
    multiindex alpha with sum n, based on a 1D node family.'''
    xn = family[n]
    b = np.zeros((d+1,))
    if d == 1:
        b[:] = xn[[alpha[0], alpha[1]]]
        return b
    weight = 0.
    for i in range(d+1):
        alpha_noti = alpha[:i] + alpha[i+1:]
        n_noti = n - alpha[i]
        w = xn[n_noti]
        br = _recursive(d-1, n_noti, alpha_noti, family)
        b[:i] += w * br[:i]
        b[i+1:] += w * br[i:]
        weight += w
    b /= weight
    return b


def _decode_family(family):
    if family is None:
        family = lgl_family
    elif isinstance(family, str):
        if family == 'lgl':
            family = lgl_family
        elif family == 'lgc':
            family = lgc_family
        elif family == 'gl':
            family = gl_family
        elif family == 'gc':
            family = gc_family
        elif family == 'equi':
            family = equi_family
        elif family == 'equi_interior':
            family = equi_interior_family
    elif isinstance(family, tuple) and len(family) == 2 and family[0] == 'lgg':
        a = family[1]
        family = NodeFamily(lambda n:
                            coord_map(lobattogaussjacobi(n+1, a-0.5, a-0.5)[0],
                                      'biunit', 'unit')
                            if n > 0 else np.array([0.5]))
    elif isinstance(family, tuple) and len(family) == 2 and family[0] == 'gg':
        a = family[1]
        family = NodeFamily(lambda n:
                            coord_map(gaussjacobi(n+1, a-0.5, a-0.5)[0],
                                      'biunit', 'unit'))
    return family


def recursive(d, n, family='lgl', domain='barycentric'):
    '''Recursively defined nodes for `\\mathcal{P}_n(\\Delta^d)`, the polynomials
    with degree at most `n` on the `d`-simplex, based on a 1D node family.

    Notes:

        Some definitions:

        - A 1D *node set* `X_k=(x_{k,0}, \\dots, x_{k,k})` is a
          sorted list of `k+1` points in `[0,1]` that is symmetric about `1/2`.

        - A 1D *node family* `\\boldsymbol{X} = \\{X_k\\}` is a collection of 1D
          node sets for every degree `k`.

        - The `barycentric triangle`_ is a canonical domain for the `d`-simplex
          in `\\mathbb{R}^{d+1}`: positive coordinates
          `\\boldsymbol{b} = (b_0, b_1, \\dots b_d)` such that
          `\\sum_i b_i = 1`.

        - For a multi-index_ `\\boldsymbol{\\alpha}`, let `\\#\\boldsymbol{\\alpha}` be its length,
          `|\\boldsymbol{\\alpha}|` its sum, and `\\boldsymbol{\\alpha}_{\\backslash i}` the multi-index
          created by removing `\\boldsymbol{\\alpha}_i`: nodes that define a basis of
          `\\mathcal{P}_n(\\Delta^d)` can be indexed by `\\boldsymbol{\\alpha}` such that
          `\\#\\boldsymbol{\\alpha} = d+1` and `|\\boldsymbol{\\alpha}| = n`.

        - For a vector `\\boldsymbol{b}`, let `\\boldsymbol{b}_{+i}` be the
          vector created by inserting a zero for the `i`-th coordinate.

        The recursive definition of barycentric node coordinates,
        `\\boldsymbol{b}_{\\boldsymbol{X}}(\\boldsymbol{\\alpha})
        \\in \\mathbb{R}^{\\#\\boldsymbol{\\alpha}}` has the base case

        .. math::
            :label: basecase

            \\boldsymbol{b}_{\\boldsymbol{X}}(\\boldsymbol{\\alpha}) =
            (x_{|\\boldsymbol{\\alpha}|,\\boldsymbol{\\alpha}_0}, x_{|\\boldsymbol{\\alpha}|,\\boldsymbol{\\alpha}_1}), \\quad
            \\#\\boldsymbol{\\alpha} = 2,

        and the recursion

        .. math::
            :label: recursion

            \\boldsymbol{b}_{\\boldsymbol{X}}(\\boldsymbol{\\alpha}) =
            \\frac{\\sum_i x_{|\\boldsymbol{\\alpha}|,|\\boldsymbol{\\alpha}_{\\backslash i}|}
            \\boldsymbol{b}_{\\boldsymbol{X}}(\\boldsymbol{\\alpha}_{\\backslash i})_{+i}}
            {\\sum_i x_{|\\boldsymbol{\\alpha}|,|\\boldsymbol{\\alpha}_{\\backslash i}|}}, \\quad
            \\#\\boldsymbol{\\alpha} > 2.

        The full set of nodes is

        .. math::
            :label: fullset

             R^d_{\\boldsymbol{X},n} =
             \\{\\boldsymbol{b}_{\\boldsymbol{X}}(\\boldsymbol{\\alpha}): \\#\\boldsymbol{\\alpha} = d+1,
             |\\boldsymbol{\\alpha}| = n\\}.

    Args:
        d (int): The dimension of the simplex
        n (int): The maximum degree of the polynomials
        family (optional): The 1D node family used to define the coordinates in
            the barycentric `d`-simplex.  The default ``family='lgl'``
            corresponds to the shifted Lobatto-Gauss-Legendre_ nodes.  See
            ":ref:`families`" for using other node families.
        domain (str, optional): The domain for the `d`-simplex where the
            returned coordinates will be defined.  See ":ref:`domains`" for the
            choices and their formal definitions.

    Returns:
        ndarray: The nodes `R^d_{\\boldsymbol{X},n}` defined in
        :eq:`fullset`, as a 2D array with `\\binom{n+d}{d}` rows.  If
        ``domain='barycentric'``, it has `d+1` columns, otherwise it has `d`
        columns.

    Examples:

        Nodes for `\\mathcal{P}^4(\\Delta^2)` in barycentric coordinates:

        >>> recursive_nodes(2, 4)
        array([[0.        , 0.        , 1.        ],
               [0.        , 0.17267316, 0.82732684],
               [0.        , 0.5       , 0.5       ],
               [0.        , 0.82732684, 0.17267316],
               [0.        , 1.        , 0.        ],
               [0.17267316, 0.        , 0.82732684],
               [0.2221552 , 0.2221552 , 0.5556896 ],
               [0.2221552 , 0.5556896 , 0.2221552 ],
               [0.17267316, 0.82732684, 0.        ],
               [0.5       , 0.        , 0.5       ],
               [0.5556896 , 0.2221552 , 0.2221552 ],
               [0.5       , 0.5       , 0.        ],
               [0.82732684, 0.        , 0.17267316],
               [0.82732684, 0.17267316, 0.        ],
               [1.        , 0.        , 0.        ]])

        The same nodes on the unit triangle:

        >>> recursive_nodes(2, 4, domain='unit')
        array([[0.        , 0.        ],
               [0.        , 0.17267316],
               [0.        , 0.5       ],
               [0.        , 0.82732684],
               [0.        , 1.        ],
               [0.17267316, 0.        ],
               [0.2221552 , 0.2221552 ],
               [0.2221552 , 0.5556896 ],
               [0.17267316, 0.82732684],
               [0.5       , 0.        ],
               [0.5556896 , 0.2221552 ],
               [0.5       , 0.5       ],
               [0.82732684, 0.        ],
               [0.82732684, 0.17267316],
               [1.        , 0.        ]])

        If we construct the node set not from the Lobatto-Gauss-Legendre 1D
        node family, but from the equispaced 1D node family, we get equispaced
        2D nodes:

        >>> recursive_nodes(2, 4, family='equi', domain='unit')
        array([[0.  , 0.  ],
               [0.  , 0.25],
               [0.  , 0.5 ],
               [0.  , 0.75],
               [0.  , 1.  ],
               [0.25, 0.  ],
               [0.25, 0.25],
               [0.25, 0.5 ],
               [0.25, 0.75],
               [0.5 , 0.  ],
               [0.5 , 0.25],
               [0.5 , 0.5 ],
               [0.75, 0.  ],
               [0.75, 0.25],
               [1.  , 0.  ]])

        This is what they look like mapped to the equilateral triangle:

        .. plot::
           :include-source: True

           >>> import matplotlib.pyplot as plt
           >>> from recursivenodes import recursive_nodes
           >>> nodes_equi = recursive_nodes(2, 4, family='equi', domain='equilateral')
           >>> nodes_lgl = recursive_nodes(2, 4, domain='equilateral')
           >>> plt.scatter(nodes_lgl[:,0], nodes_lgl[:,1], marker='o', label='recursive LGL')
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.scatter(nodes_equi[:,0], nodes_equi[:,1], marker='^', label='equispaced')
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.gca().set_aspect('equal')
           >>> plt.legend()
           <matplotlib.legend.Legend object at ...>
           >>> plt.show()


    .. _multi-index: https://en.wikipedia.org/wiki/Multi-index_notation
    .. _barycentric triangle: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    .. _Lobatto-Gauss-Legendre: https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Lobatto_rules

    '''
    family = _decode_family(family)
    N = npolys(d, n)
    x = np.zeros((N, d+1))
    for (k, i) in enumerate(multiindex_equal(d+1, n)):
        x[k, :] = _recursive(d, n, i, family)
    return coord_map(x, 'barycentric', domain)


# ---------------------------START---------------------------- #
# Rapetti, Sommariva, & Vianello Symmetric Lebesgue-minimizing #
# Lobatto-Gauss-Legendre nodes: doi:10.1016/j.cam.2011.11.023  #

_rsv_lebgls = {}

_rsv_lebgls[1] = [
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        ]

_rsv_lebgls[2] = [
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [5.000000000000000e-01, 5.000000000000000e-01],
        [5.000000000000000e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 5.000000000000000e-01],
        ]

_rsv_lebgls[3] = [
        [3.333333333333333e-01, 3.333333333333333e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [2.763932022500210e-01, 7.236067977499789e-01],
        [7.236067977499789e-01, 2.763932022500210e-01],
        [2.763932022500210e-01, 0.000000000000000e+00],
        [7.236067977499789e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 2.763932022500210e-01],
        [0.000000000000000e+00, 7.236067977499789e-01],
        ]

_rsv_lebgls[4] = [
        [2.371066302371978e-01, 2.371066302371978e-01],
        [2.371066302371978e-01, 5.257867395256041e-01],
        [5.257867395256041e-01, 2.371066302371978e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.726731646460114e-01, 8.273268353539885e-01],
        [5.000000000000000e-01, 5.000000000000000e-01],
        [8.273268353539885e-01, 1.726731646460114e-01],
        [1.726731646460114e-01, 0.000000000000000e+00],
        [5.000000000000000e-01, 0.000000000000000e+00],
        [8.273268353539885e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.726731646460114e-01],
        [0.000000000000000e+00, 5.000000000000000e-01],
        [0.000000000000000e+00, 8.273268353539885e-01],
        ]

_rsv_lebgls[5] = [
        [4.123406940197869e-01, 4.123406940197869e-01],
        [1.737985196489401e-01, 1.737985196489401e-01],
        [4.123406940197869e-01, 1.753186119604261e-01],
        [1.737985196489401e-01, 6.524029607021195e-01],
        [1.753186119604261e-01, 4.123406940197869e-01],
        [6.524029607021195e-01, 1.737985196489401e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.174723380352677e-01, 8.825276619647322e-01],
        [3.573842417596774e-01, 6.426157582403224e-01],
        [6.426157582403225e-01, 3.573842417596774e-01],
        [8.825276619647324e-01, 1.174723380352675e-01],
        [1.174723380352677e-01, 0.000000000000000e+00],
        [3.573842417596774e-01, 0.000000000000000e+00],
        [6.426157582403225e-01, 0.000000000000000e+00],
        [8.825276619647324e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.174723380352677e-01],
        [0.000000000000000e+00, 3.573842417596774e-01],
        [0.000000000000000e+00, 6.426157582403225e-01],
        [0.000000000000000e+00, 8.825276619647324e-01],
        ]

_rsv_lebgls[6] = [
        [3.333333333333333e-01, 3.333333333333333e-01],
        [1.236700206266602e-01, 1.236700206266602e-01],
        [1.236700206266602e-01, 7.526599587466795e-01],
        [7.526599587466795e-01, 1.236700206266602e-01],
        [3.254637175704040e-01, 5.425665225375484e-01],
        [3.254637175704040e-01, 1.319697598920475e-01],
        [5.425665225375484e-01, 3.254637175704040e-01],
        [5.425665225375484e-01, 1.319697598920475e-01],
        [1.319697598920475e-01, 3.254637175704040e-01],
        [1.319697598920475e-01, 5.425665225375484e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [8.488805186071646e-02, 9.151119481392835e-01],
        [2.655756032646428e-01, 7.344243967353571e-01],
        [5.000000000000001e-01, 4.999999999999998e-01],
        [7.344243967353571e-01, 2.655756032646428e-01],
        [9.151119481392833e-01, 8.488805186071668e-02],
        [8.488805186071646e-02, 0.000000000000000e+00],
        [2.655756032646428e-01, 0.000000000000000e+00],
        [5.000000000000001e-01, 0.000000000000000e+00],
        [7.344243967353571e-01, 0.000000000000000e+00],
        [9.151119481392833e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 8.488805186071646e-02],
        [0.000000000000000e+00, 2.655756032646428e-01],
        [0.000000000000000e+00, 5.000000000000001e-01],
        [0.000000000000000e+00, 7.344243967353571e-01],
        [0.000000000000000e+00, 9.151119481392833e-01],
        ]

_rsv_lebgls[7] = [
        [2.741584394628903e-01, 2.741584394628903e-01],
        [8.791461398121544e-02, 8.791461398121544e-02],
        [2.741584394628903e-01, 4.516831210742193e-01],
        [8.791461398121544e-02, 8.241707720375690e-01],
        [4.516831210742193e-01, 2.741584394628903e-01],
        [8.241707720375690e-01, 8.791461398121544e-02],
        [4.500074460521015e-01, 4.500074460521015e-01],
        [4.500074460521015e-01, 9.998510789579695e-02],
        [9.998510789579695e-02, 4.500074460521015e-01],
        [2.574575396908898e-01, 6.389182038566255e-01],
        [2.574575396908898e-01, 1.036242564524846e-01],
        [6.389182038566255e-01, 2.574575396908898e-01],
        [6.389182038566255e-01, 1.036242564524846e-01],
        [1.036242564524846e-01, 2.574575396908898e-01],
        [1.036242564524846e-01, 6.389182038566255e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [6.412992574519671e-02, 9.358700742548032e-01],
        [2.041499092834289e-01, 7.958500907165710e-01],
        [3.953503910487606e-01, 6.046496089512394e-01],
        [6.046496089512393e-01, 3.953503910487606e-01],
        [7.958500907165713e-01, 2.041499092834286e-01],
        [9.358700742548031e-01, 6.412992574519682e-02],
        [6.412992574519671e-02, 0.000000000000000e+00],
        [2.041499092834289e-01, 0.000000000000000e+00],
        [3.953503910487606e-01, 0.000000000000000e+00],
        [6.046496089512393e-01, 0.000000000000000e+00],
        [7.958500907165713e-01, 0.000000000000000e+00],
        [9.358700742548031e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 6.412992574519671e-02],
        [0.000000000000000e+00, 2.041499092834289e-01],
        [0.000000000000000e+00, 3.953503910487606e-01],
        [0.000000000000000e+00, 6.046496089512393e-01],
        [0.000000000000000e+00, 7.958500907165713e-01],
        [0.000000000000000e+00, 9.358700742548031e-01],
        ]

_rsv_lebgls[8] = [
        [2.238114422946618e-01, 2.238114422946618e-01],
        [3.879143834569392e-01, 3.879143834569392e-01],
        [6.749944697564996e-02, 6.749944697564996e-02],
        [2.238114422946618e-01, 5.523771154106762e-01],
        [3.879143834569392e-01, 2.241712330861213e-01],
        [6.749944697564996e-02, 8.650011060487001e-01],
        [5.523771154106762e-01, 2.238114422946618e-01],
        [2.241712330861213e-01, 3.879143834569392e-01],
        [8.650011060487001e-01, 6.749944697564996e-02],
        [2.106241399947925e-01, 7.064768674963645e-01],
        [3.702480264146488e-01, 5.482896391715870e-01],
        [2.106241399947925e-01, 8.289899250884291e-02],
        [3.702480264146488e-01, 8.146233441376415e-02],
        [7.064768674963645e-01, 2.106241399947925e-01],
        [5.482896391715870e-01, 3.702480264146488e-01],
        [7.064768674963645e-01, 8.289899250884291e-02],
        [5.482896391715870e-01, 8.146233441376415e-02],
        [8.289899250884291e-02, 2.106241399947925e-01],
        [8.146233441376415e-02, 3.702480264146488e-01],
        [8.289899250884291e-02, 7.064768674963645e-01],
        [8.146233441376415e-02, 5.482896391715870e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [5.012100229426996e-02, 9.498789977057300e-01],
        [1.614068602446310e-01, 8.385931397553689e-01],
        [3.184412680869108e-01, 6.815587319130891e-01],
        [5.000000000000001e-01, 4.999999999999998e-01],
        [6.815587319130891e-01, 3.184412680869108e-01],
        [8.385931397553689e-01, 1.614068602446310e-01],
        [9.498789977057298e-01, 5.012100229427018e-02],
        [5.012100229426996e-02, 0.000000000000000e+00],
        [1.614068602446310e-01, 0.000000000000000e+00],
        [3.184412680869108e-01, 0.000000000000000e+00],
        [5.000000000000001e-01, 0.000000000000000e+00],
        [6.815587319130891e-01, 0.000000000000000e+00],
        [8.385931397553689e-01, 0.000000000000000e+00],
        [9.498789977057298e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 5.012100229426996e-02],
        [0.000000000000000e+00, 1.614068602446310e-01],
        [0.000000000000000e+00, 3.184412680869108e-01],
        [0.000000000000000e+00, 5.000000000000001e-01],
        [0.000000000000000e+00, 6.815587319130891e-01],
        [0.000000000000000e+00, 8.385931397553689e-01],
        [0.000000000000000e+00, 9.498789977057298e-01],
        ]

_rsv_lebgls[9] = [
        [3.333333333333333e-01, 3.333333333333333e-01],
        [5.208382829746331e-02, 5.208382829746331e-02],
        [1.835842328020949e-01, 1.835842328020949e-01],
        [5.208382829746331e-02, 8.958323434050733e-01],
        [1.835842328020949e-01, 6.328315343958100e-01],
        [8.958323434050733e-01, 5.208382829746331e-02],
        [6.328315343958100e-01, 1.835842328020949e-01],
        [4.677778353675260e-01, 4.677778353675260e-01],
        [4.677778353675260e-01, 6.444432926494780e-02],
        [6.444432926494780e-02, 4.677778353675260e-01],
        [3.335906038202737e-01, 4.796482965082551e-01],
        [1.723053623095807e-01, 7.617793422357387e-01],
        [3.093824727565271e-01, 6.225829731720886e-01],
        [3.335906038202737e-01, 1.867610996714711e-01],
        [1.723053623095807e-01, 6.591529545468055e-02],
        [3.093824727565271e-01, 6.803455407138414e-02],
        [4.796482965082551e-01, 3.335906038202737e-01],
        [7.617793422357387e-01, 1.723053623095807e-01],
        [6.225829731720886e-01, 3.093824727565271e-01],
        [4.796482965082551e-01, 1.867610996714711e-01],
        [7.617793422357387e-01, 6.591529545468055e-02],
        [6.225829731720886e-01, 6.803455407138414e-02],
        [1.867610996714711e-01, 3.335906038202737e-01],
        [6.591529545468055e-02, 1.723053623095807e-01],
        [6.803455407138414e-02, 3.093824727565271e-01],
        [1.867610996714711e-01, 4.796482965082551e-01],
        [6.591529545468055e-02, 7.617793422357387e-01],
        [6.803455407138414e-02, 6.225829731720886e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [4.023304591677051e-02, 9.597669540832294e-01],
        [1.306130674472475e-01, 8.693869325527523e-01],
        [2.610375250947778e-01, 7.389624749052221e-01],
        [4.173605211668064e-01, 5.826394788331934e-01],
        [5.826394788331933e-01, 4.173605211668066e-01],
        [7.389624749052223e-01, 2.610375250947776e-01],
        [8.693869325527523e-01, 1.306130674472476e-01],
        [9.597669540832292e-01, 4.023304591677079e-02],
        [4.023304591677051e-02, 0.000000000000000e+00],
        [1.306130674472475e-01, 0.000000000000000e+00],
        [2.610375250947778e-01, 0.000000000000000e+00],
        [4.173605211668064e-01, 0.000000000000000e+00],
        [5.826394788331933e-01, 0.000000000000000e+00],
        [7.389624749052223e-01, 0.000000000000000e+00],
        [8.693869325527523e-01, 0.000000000000000e+00],
        [9.597669540832292e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 4.023304591677051e-02],
        [0.000000000000000e+00, 1.306130674472475e-01],
        [0.000000000000000e+00, 2.610375250947778e-01],
        [0.000000000000000e+00, 4.173605211668064e-01],
        [0.000000000000000e+00, 5.826394788331933e-01],
        [0.000000000000000e+00, 7.389624749052223e-01],
        [0.000000000000000e+00, 8.693869325527523e-01],
        [0.000000000000000e+00, 9.597669540832292e-01],
        ]

_rsv_lebgls[10] = [
        [1.312733767493933e-01, 1.312733767493933e-01],
        [5.386953019930516e-02, 5.386953019930516e-02],
        [2.252378431088206e-01, 2.252378431088206e-01],
        [1.312733767493933e-01, 7.374532465012133e-01],
        [5.386953019930516e-02, 8.922609396013896e-01],
        [2.252378431088206e-01, 5.495243137823585e-01],
        [7.374532465012133e-01, 1.312733767493933e-01],
        [8.922609396013896e-01, 5.386953019930516e-02],
        [5.495243137823585e-01, 2.252378431088206e-01],
        [3.824877939073025e-01, 3.824877939073025e-01],
        [3.824877939073025e-01, 2.350244121853949e-01],
        [2.350244121853949e-01, 3.824877939073025e-01],
        [2.622175009707391e-01, 6.564279951505325e-01],
        [3.776961071421849e-01, 5.223502436193665e-01],
        [1.820746613840211e-01, 7.792454007990795e-01],
        [3.779628077066509e-01, 5.892534012902733e-01],
        [2.622175009707391e-01, 8.135450387872833e-02],
        [3.776961071421849e-01, 9.995364923844851e-02],
        [1.820746613840211e-01, 3.867993781689926e-02],
        [3.779628077066509e-01, 3.278379100307571e-02],
        [6.564279951505325e-01, 2.622175009707391e-01],
        [5.223502436193665e-01, 3.776961071421849e-01],
        [7.792454007990795e-01, 1.820746613840211e-01],
        [5.892534012902733e-01, 3.779628077066509e-01],
        [6.564279951505325e-01, 8.135450387872833e-02],
        [5.223502436193665e-01, 9.995364923844851e-02],
        [7.792454007990795e-01, 3.867993781689926e-02],
        [5.892534012902733e-01, 3.278379100307571e-02],
        [8.135450387872833e-02, 2.622175009707391e-01],
        [9.995364923844851e-02, 3.776961071421849e-01],
        [3.867993781689926e-02, 1.820746613840211e-01],
        [3.278379100307571e-02, 3.779628077066509e-01],
        [8.135450387872833e-02, 6.564279951505325e-01],
        [9.995364923844851e-02, 5.223502436193665e-01],
        [3.867993781689926e-02, 7.792454007990795e-01],
        [3.278379100307571e-02, 5.892534012902733e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [3.299928479597025e-02, 9.670007152040297e-01],
        [1.077582631684279e-01, 8.922417368315720e-01],
        [2.173823365018974e-01, 7.826176634981025e-01],
        [3.521209322065304e-01, 6.478790677934695e-01],
        [5.000000000000000e-01, 5.000000000000000e-01],
        [6.478790677934695e-01, 3.521209322065304e-01],
        [7.826176634981024e-01, 2.173823365018975e-01],
        [8.922417368315722e-01, 1.077582631684277e-01],
        [9.670007152040296e-01, 3.299928479597036e-02],
        [3.299928479597025e-02, 0.000000000000000e+00],
        [1.077582631684279e-01, 0.000000000000000e+00],
        [2.173823365018974e-01, 0.000000000000000e+00],
        [3.521209322065304e-01, 0.000000000000000e+00],
        [5.000000000000000e-01, 0.000000000000000e+00],
        [6.478790677934695e-01, 0.000000000000000e+00],
        [7.826176634981024e-01, 0.000000000000000e+00],
        [8.922417368315722e-01, 0.000000000000000e+00],
        [9.670007152040296e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 3.299928479597025e-02],
        [0.000000000000000e+00, 1.077582631684279e-01],
        [0.000000000000000e+00, 2.173823365018974e-01],
        [0.000000000000000e+00, 3.521209322065304e-01],
        [0.000000000000000e+00, 5.000000000000000e-01],
        [0.000000000000000e+00, 6.478790677934695e-01],
        [0.000000000000000e+00, 7.826176634981024e-01],
        [0.000000000000000e+00, 8.922417368315722e-01],
        [0.000000000000000e+00, 9.670007152040296e-01],
        ]

_rsv_lebgls[11] = [
        [1.050821229581930e-01, 1.050821229581930e-01],
        [4.295650634280591e-02, 4.295650634280591e-02],
        [2.195219638878813e-01, 2.195219638878813e-01],
        [1.050821229581930e-01, 7.898357540836139e-01],
        [4.295650634280591e-02, 9.140869873143881e-01],
        [2.195219638878813e-01, 5.609560722242372e-01],
        [7.898357540836139e-01, 1.050821229581930e-01],
        [9.140869873143881e-01, 4.295650634280591e-02],
        [5.609560722242372e-01, 2.195219638878813e-01],
        [4.807601837574821e-01, 4.807601837574821e-01],
        [4.286809903071824e-01, 4.286809903071824e-01],
        [4.807601837574821e-01, 3.847963248503571e-02],
        [4.286809903071824e-01, 1.426380193856351e-01],
        [3.847963248503571e-02, 4.807601837574821e-01],
        [1.426380193856351e-01, 4.286809903071824e-01],
        [1.994095640340481e-01, 7.069562305387490e-01],
        [1.474224961654246e-01, 8.177875722890196e-01],
        [3.075960627730014e-01, 6.626688579526240e-01],
        [3.278085121320318e-01, 4.251975363710968e-01],
        [3.205222948140067e-01, 5.840805307965026e-01],
        [1.994095640340481e-01, 9.363420542720279e-02],
        [1.474224961654246e-01, 3.478993154555565e-02],
        [3.075960627730014e-01, 2.973507927437446e-02],
        [3.278085121320318e-01, 2.469939514968714e-01],
        [3.205222948140067e-01, 9.539717438949057e-02],
        [7.069562305387490e-01, 1.994095640340481e-01],
        [8.177875722890196e-01, 1.474224961654246e-01],
        [6.626688579526240e-01, 3.075960627730014e-01],
        [4.251975363710968e-01, 3.278085121320318e-01],
        [5.840805307965026e-01, 3.205222948140067e-01],
        [7.069562305387490e-01, 9.363420542720279e-02],
        [8.177875722890196e-01, 3.478993154555565e-02],
        [6.626688579526240e-01, 2.973507927437446e-02],
        [4.251975363710968e-01, 2.469939514968714e-01],
        [5.840805307965026e-01, 9.539717438949057e-02],
        [9.363420542720279e-02, 1.994095640340481e-01],
        [3.478993154555565e-02, 1.474224961654246e-01],
        [2.973507927437446e-02, 3.075960627730014e-01],
        [2.469939514968714e-01, 3.278085121320318e-01],
        [9.539717438949057e-02, 3.205222948140067e-01],
        [9.363420542720279e-02, 7.069562305387490e-01],
        [3.478993154555565e-02, 8.177875722890196e-01],
        [2.973507927437446e-02, 6.626688579526240e-01],
        [2.469939514968714e-01, 4.251975363710968e-01],
        [9.539717438949057e-02, 5.840805307965026e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [2.755036388855880e-02, 9.724496361114411e-01],
        [9.036033917799679e-02, 9.096396608220032e-01],
        [1.835619234840696e-01, 8.164380765159302e-01],
        [3.002345295173255e-01, 6.997654704826744e-01],
        [4.317235335725362e-01, 5.682764664274637e-01],
        [5.682764664274635e-01, 4.317235335725364e-01],
        [6.997654704826742e-01, 3.002345295173257e-01],
        [8.164380765159302e-01, 1.835619234840697e-01],
        [9.096396608220034e-01, 9.036033917799657e-02],
        [9.724496361114410e-01, 2.755036388855891e-02],
        [2.755036388855880e-02, 0.000000000000000e+00],
        [9.036033917799679e-02, 0.000000000000000e+00],
        [1.835619234840696e-01, 0.000000000000000e+00],
        [3.002345295173255e-01, 0.000000000000000e+00],
        [4.317235335725362e-01, 0.000000000000000e+00],
        [5.682764664274635e-01, 0.000000000000000e+00],
        [6.997654704826742e-01, 0.000000000000000e+00],
        [8.164380765159302e-01, 0.000000000000000e+00],
        [9.096396608220034e-01, 0.000000000000000e+00],
        [9.724496361114410e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 2.755036388855880e-02],
        [0.000000000000000e+00, 9.036033917799679e-02],
        [0.000000000000000e+00, 1.835619234840696e-01],
        [0.000000000000000e+00, 3.002345295173255e-01],
        [0.000000000000000e+00, 4.317235335725362e-01],
        [0.000000000000000e+00, 5.682764664274635e-01],
        [0.000000000000000e+00, 6.997654704826742e-01],
        [0.000000000000000e+00, 8.164380765159302e-01],
        [0.000000000000000e+00, 9.096396608220034e-01],
        [0.000000000000000e+00, 9.724496361114410e-01],
        ]

_rsv_lebgls[12] = [
        [3.333333333333333e-01, 3.333333333333333e-01],
        [4.108337681828540e-02, 4.108337681828540e-02],
        [2.530162527885500e-01, 2.530162527885500e-01],
        [9.275221949490589e-02, 9.275221949490589e-02],
        [4.108337681828540e-02, 9.178332463634292e-01],
        [2.530162527885500e-01, 4.939674944228998e-01],
        [9.275221949490589e-02, 8.144955610101881e-01],
        [9.178332463634292e-01, 4.108337681828540e-02],
        [4.939674944228998e-01, 2.530162527885500e-01],
        [8.144955610101881e-01, 9.275221949490589e-02],
        [4.044590487503580e-01, 4.044590487503580e-01],
        [4.044590487503580e-01, 1.910819024992839e-01],
        [1.910819024992839e-01, 4.044590487503580e-01],
        [2.815910345568075e-01, 6.409342565390805e-01],
        [4.080479136914563e-01, 5.518892594390213e-01],
        [1.791741767343328e-01, 7.271665681840357e-01],
        [2.625999784981004e-01, 7.120109530779203e-01],
        [3.930958962903537e-01, 4.966440289045495e-01],
        [2.367253835467234e-01, 6.047812780184623e-01],
        [1.274088629870740e-01, 8.437470933281366e-01],
        [2.815910345568075e-01, 7.747470890411189e-02],
        [4.080479136914563e-01, 4.006282686952233e-02],
        [1.791741767343328e-01, 9.365925508163130e-02],
        [2.625999784981004e-01, 2.538906842397914e-02],
        [3.930958962903537e-01, 1.102600748050966e-01],
        [2.367253835467234e-01, 1.584933384348142e-01],
        [1.274088629870740e-01, 2.884404368478932e-02],
        [6.409342565390805e-01, 2.815910345568075e-01],
        [5.518892594390213e-01, 4.080479136914563e-01],
        [7.271665681840357e-01, 1.791741767343328e-01],
        [7.120109530779203e-01, 2.625999784981004e-01],
        [4.966440289045495e-01, 3.930958962903537e-01],
        [6.047812780184623e-01, 2.367253835467234e-01],
        [8.437470933281366e-01, 1.274088629870740e-01],
        [6.409342565390805e-01, 7.747470890411189e-02],
        [5.518892594390213e-01, 4.006282686952233e-02],
        [7.271665681840357e-01, 9.365925508163130e-02],
        [7.120109530779203e-01, 2.538906842397914e-02],
        [4.966440289045495e-01, 1.102600748050966e-01],
        [6.047812780184623e-01, 1.584933384348142e-01],
        [8.437470933281366e-01, 2.884404368478932e-02],
        [7.747470890411189e-02, 2.815910345568075e-01],
        [4.006282686952233e-02, 4.080479136914563e-01],
        [9.365925508163130e-02, 1.791741767343328e-01],
        [2.538906842397914e-02, 2.625999784981004e-01],
        [1.102600748050966e-01, 3.930958962903537e-01],
        [1.584933384348142e-01, 2.367253835467234e-01],
        [2.884404368478932e-02, 1.274088629870740e-01],
        [7.747470890411189e-02, 6.409342565390805e-01],
        [4.006282686952233e-02, 5.518892594390213e-01],
        [9.365925508163130e-02, 7.271665681840357e-01],
        [2.538906842397914e-02, 7.120109530779203e-01],
        [1.102600748050966e-01, 4.966440289045495e-01],
        [1.584933384348142e-01, 6.047812780184623e-01],
        [2.884404368478932e-02, 8.437470933281366e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [2.334507667891805e-02, 9.766549233210819e-01],
        [7.682621767406377e-02, 9.231737823259362e-01],
        [1.569057654591214e-01, 8.430942345408785e-01],
        [2.585450894543320e-01, 7.414549105456679e-01],
        [3.753565349468798e-01, 6.246434650531200e-01],
        [4.999999999999998e-01, 5.000000000000001e-01],
        [6.246434650531199e-01, 3.753565349468800e-01],
        [7.414549105456680e-01, 2.585450894543319e-01],
        [8.430942345408788e-01, 1.569057654591211e-01],
        [9.231737823259358e-01, 7.682621767406416e-02],
        [9.766549233210819e-01, 2.334507667891805e-02],
        [2.334507667891805e-02, 0.000000000000000e+00],
        [7.682621767406377e-02, 0.000000000000000e+00],
        [1.569057654591214e-01, 0.000000000000000e+00],
        [2.585450894543320e-01, 0.000000000000000e+00],
        [3.753565349468798e-01, 0.000000000000000e+00],
        [4.999999999999998e-01, 0.000000000000000e+00],
        [6.246434650531199e-01, 0.000000000000000e+00],
        [7.414549105456680e-01, 0.000000000000000e+00],
        [8.430942345408788e-01, 0.000000000000000e+00],
        [9.231737823259358e-01, 0.000000000000000e+00],
        [9.766549233210819e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 2.334507667891805e-02],
        [0.000000000000000e+00, 7.682621767406377e-02],
        [0.000000000000000e+00, 1.569057654591214e-01],
        [0.000000000000000e+00, 2.585450894543320e-01],
        [0.000000000000000e+00, 3.753565349468798e-01],
        [0.000000000000000e+00, 4.999999999999998e-01],
        [0.000000000000000e+00, 6.246434650531199e-01],
        [0.000000000000000e+00, 7.414549105456680e-01],
        [0.000000000000000e+00, 8.430942345408788e-01],
        [0.000000000000000e+00, 9.231737823259358e-01],
        [0.000000000000000e+00, 9.766549233210819e-01],
        ]

_rsv_lebgls[13] = [
        [1.480452652753508e-01, 1.480452652753508e-01],
        [8.377435391383915e-02, 8.377435391383915e-02],
        [3.446157135859970e-02, 3.446157135859970e-02],
        [1.480452652753508e-01, 7.039094694492982e-01],
        [8.377435391383915e-02, 8.324512921723217e-01],
        [3.446157135859970e-02, 9.310768572828006e-01],
        [7.039094694492982e-01, 1.480452652753508e-01],
        [8.324512921723217e-01, 8.377435391383915e-02],
        [9.310768572828006e-01, 3.446157135859970e-02],
        [4.655749386684967e-01, 4.655749386684967e-01],
        [3.658220186966510e-01, 3.658220186966510e-01],
        [4.308181049840558e-01, 4.308181049840558e-01],
        [4.655749386684967e-01, 6.885012266300649e-02],
        [3.658220186966510e-01, 2.683559626066979e-01],
        [4.308181049840558e-01, 1.383637900318882e-01],
        [6.885012266300649e-02, 4.655749386684967e-01],
        [2.683559626066979e-01, 3.658220186966510e-01],
        [1.383637900318882e-01, 4.308181049840558e-01],
        [1.876198296639073e-01, 7.505363476334480e-01],
        [2.408988278555256e-01, 6.203711602337613e-01],
        [1.095911259954575e-01, 8.671079494311813e-01],
        [3.011955592096039e-01, 6.516018514940782e-01],
        [2.401186982114374e-01, 7.423494369880164e-01],
        [4.086944595865063e-01, 5.728822400760038e-01],
        [2.755611410714366e-01, 5.088600522034840e-01],
        [3.311895891298100e-01, 5.665822045660098e-01],
        [1.876198296639073e-01, 6.184382270264465e-02],
        [2.408988278555256e-01, 1.387300119107129e-01],
        [1.095911259954575e-01, 2.330092457336119e-02],
        [3.011955592096039e-01, 4.720258929631782e-02],
        [2.401186982114374e-01, 1.753186480054613e-02],
        [4.086944595865063e-01, 1.842330033748984e-02],
        [2.755611410714366e-01, 2.155788067250792e-01],
        [3.311895891298100e-01, 1.022282063041801e-01],
        [7.505363476334480e-01, 1.876198296639073e-01],
        [6.203711602337613e-01, 2.408988278555256e-01],
        [8.671079494311813e-01, 1.095911259954575e-01],
        [6.516018514940782e-01, 3.011955592096039e-01],
        [7.423494369880164e-01, 2.401186982114374e-01],
        [5.728822400760038e-01, 4.086944595865063e-01],
        [5.088600522034840e-01, 2.755611410714366e-01],
        [5.665822045660098e-01, 3.311895891298100e-01],
        [7.505363476334480e-01, 6.184382270264465e-02],
        [6.203711602337613e-01, 1.387300119107129e-01],
        [8.671079494311813e-01, 2.330092457336119e-02],
        [6.516018514940782e-01, 4.720258929631782e-02],
        [7.423494369880164e-01, 1.753186480054613e-02],
        [5.728822400760038e-01, 1.842330033748984e-02],
        [5.088600522034840e-01, 2.155788067250792e-01],
        [5.665822045660098e-01, 1.022282063041801e-01],
        [6.184382270264465e-02, 1.876198296639073e-01],
        [1.387300119107129e-01, 2.408988278555256e-01],
        [2.330092457336119e-02, 1.095911259954575e-01],
        [4.720258929631782e-02, 3.011955592096039e-01],
        [1.753186480054613e-02, 2.401186982114374e-01],
        [1.842330033748984e-02, 4.086944595865063e-01],
        [2.155788067250792e-01, 2.755611410714366e-01],
        [1.022282063041801e-01, 3.311895891298100e-01],
        [6.184382270264465e-02, 7.505363476334480e-01],
        [1.387300119107129e-01, 6.203711602337613e-01],
        [2.330092457336119e-02, 8.671079494311813e-01],
        [4.720258929631782e-02, 6.516018514940782e-01],
        [1.753186480054613e-02, 7.423494369880164e-01],
        [1.842330033748984e-02, 5.728822400760038e-01],
        [2.155788067250792e-01, 5.088600522034840e-01],
        [1.022282063041801e-01, 5.665822045660098e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [2.003247736636959e-02, 9.799675226336304e-01],
        [6.609947308482655e-02, 9.339005269151734e-01],
        [1.355657004543369e-01, 8.644342995456630e-01],
        [2.246802985356765e-01, 7.753197014643233e-01],
        [3.286379933286436e-01, 6.713620066713563e-01],
        [4.418340655581481e-01, 5.581659344418519e-01],
        [5.581659344418519e-01, 4.418340655581480e-01],
        [6.713620066713563e-01, 3.286379933286436e-01],
        [7.753197014643234e-01, 2.246802985356765e-01],
        [8.644342995456628e-01, 1.355657004543371e-01],
        [9.339005269151735e-01, 6.609947308482644e-02],
        [9.799675226336305e-01, 2.003247736636948e-02],
        [2.003247736636959e-02, 0.000000000000000e+00],
        [6.609947308482655e-02, 0.000000000000000e+00],
        [1.355657004543369e-01, 0.000000000000000e+00],
        [2.246802985356765e-01, 0.000000000000000e+00],
        [3.286379933286436e-01, 0.000000000000000e+00],
        [4.418340655581481e-01, 0.000000000000000e+00],
        [5.581659344418519e-01, 0.000000000000000e+00],
        [6.713620066713563e-01, 0.000000000000000e+00],
        [7.753197014643234e-01, 0.000000000000000e+00],
        [8.644342995456628e-01, 0.000000000000000e+00],
        [9.339005269151735e-01, 0.000000000000000e+00],
        [9.799675226336305e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 2.003247736636959e-02],
        [0.000000000000000e+00, 6.609947308482655e-02],
        [0.000000000000000e+00, 1.355657004543369e-01],
        [0.000000000000000e+00, 2.246802985356765e-01],
        [0.000000000000000e+00, 3.286379933286436e-01],
        [0.000000000000000e+00, 4.418340655581481e-01],
        [0.000000000000000e+00, 5.581659344418519e-01],
        [0.000000000000000e+00, 6.713620066713563e-01],
        [0.000000000000000e+00, 7.753197014643234e-01],
        [0.000000000000000e+00, 8.644342995456628e-01],
        [0.000000000000000e+00, 9.339005269151735e-01],
        [0.000000000000000e+00, 9.799675226336305e-01],
        ]

_rsv_lebgls[14] = [
        [2.480357184193379e-01, 2.480357184193379e-01],
        [1.456863552793877e-01, 1.456863552793877e-01],
        [6.329150383943313e-02, 6.329150383943313e-02],
        [2.529465459854143e-02, 2.529465459854143e-02],
        [2.480357184193379e-01, 5.039285631613240e-01],
        [1.456863552793877e-01, 7.086272894412243e-01],
        [6.329150383943313e-02, 8.734169923211337e-01],
        [2.529465459854143e-02, 9.494106908029170e-01],
        [5.039285631613240e-01, 2.480357184193379e-01],
        [7.086272894412243e-01, 1.456863552793877e-01],
        [8.734169923211337e-01, 6.329150383943313e-02],
        [9.494106908029170e-01, 2.529465459854143e-02],
        [4.144353019078003e-01, 4.144353019078003e-01],
        [4.581970243973809e-01, 4.581970243973809e-01],
        [4.144353019078003e-01, 1.711293961843993e-01],
        [4.581970243973809e-01, 8.360595120523806e-02],
        [1.711293961843993e-01, 4.144353019078003e-01],
        [8.360595120523806e-02, 4.581970243973809e-01],
        [2.328433256200428e-01, 6.207971314222264e-01],
        [3.356649442686338e-01, 5.253621216168752e-01],
        [2.173235932081969e-01, 7.084887848223708e-01],
        [3.360380238642644e-01, 5.889993254408694e-01],
        [3.168899942468522e-01, 6.569340955504540e-01],
        [1.269728328340639e-01, 8.095167697311791e-01],
        [9.346022651957634e-02, 8.882284430310810e-01],
        [2.011513660730161e-01, 7.745114061393802e-01],
        [3.244865857496181e-01, 4.096454737120259e-01],
        [4.293343266786438e-01, 5.406727865350026e-01],
        [2.328433256200428e-01, 1.463595429577306e-01],
        [3.356649442686338e-01, 1.389729341144909e-01],
        [2.173235932081969e-01, 7.418762196943218e-02],
        [3.360380238642644e-01, 7.496265069486618e-02],
        [3.168899942468522e-01, 2.617591020269371e-02],
        [1.269728328340639e-01, 6.351039743475683e-02],
        [9.346022651957634e-02, 1.831133044934263e-02],
        [2.011513660730161e-01, 2.433722778760361e-02],
        [3.244865857496181e-01, 2.658679405383558e-01],
        [4.293343266786438e-01, 2.999288678635358e-02],
        [6.207971314222264e-01, 2.328433256200428e-01],
        [5.253621216168752e-01, 3.356649442686338e-01],
        [7.084887848223708e-01, 2.173235932081969e-01],
        [5.889993254408694e-01, 3.360380238642644e-01],
        [6.569340955504540e-01, 3.168899942468522e-01],
        [8.095167697311791e-01, 1.269728328340639e-01],
        [8.882284430310810e-01, 9.346022651957634e-02],
        [7.745114061393802e-01, 2.011513660730161e-01],
        [4.096454737120259e-01, 3.244865857496181e-01],
        [5.406727865350026e-01, 4.293343266786438e-01],
        [6.207971314222264e-01, 1.463595429577306e-01],
        [5.253621216168752e-01, 1.389729341144909e-01],
        [7.084887848223708e-01, 7.418762196943218e-02],
        [5.889993254408694e-01, 7.496265069486618e-02],
        [6.569340955504540e-01, 2.617591020269371e-02],
        [8.095167697311791e-01, 6.351039743475683e-02],
        [8.882284430310810e-01, 1.831133044934263e-02],
        [7.745114061393802e-01, 2.433722778760361e-02],
        [4.096454737120259e-01, 2.658679405383558e-01],
        [5.406727865350026e-01, 2.999288678635358e-02],
        [1.463595429577306e-01, 2.328433256200428e-01],
        [1.389729341144909e-01, 3.356649442686338e-01],
        [7.418762196943218e-02, 2.173235932081969e-01],
        [7.496265069486618e-02, 3.360380238642644e-01],
        [2.617591020269371e-02, 3.168899942468522e-01],
        [6.351039743475683e-02, 1.269728328340639e-01],
        [1.831133044934263e-02, 9.346022651957634e-02],
        [2.433722778760361e-02, 2.011513660730161e-01],
        [2.658679405383558e-01, 3.244865857496181e-01],
        [2.999288678635358e-02, 4.293343266786438e-01],
        [1.463595429577306e-01, 6.207971314222264e-01],
        [1.389729341144909e-01, 5.253621216168752e-01],
        [7.418762196943218e-02, 7.084887848223708e-01],
        [7.496265069486618e-02, 5.889993254408694e-01],
        [2.617591020269371e-02, 6.569340955504540e-01],
        [6.351039743475683e-02, 8.095167697311791e-01],
        [1.831133044934263e-02, 8.882284430310810e-01],
        [2.433722778760361e-02, 7.745114061393802e-01],
        [2.658679405383558e-01, 4.096454737120259e-01],
        [2.999288678635358e-02, 5.406727865350026e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.737703674808060e-02, 9.826229632519194e-01],
        [5.745897788851178e-02, 9.425410221114882e-01],
        [1.182401550240923e-01, 8.817598449759076e-01],
        [1.968733972650772e-01, 8.031266027349226e-01],
        [2.896809726431638e-01, 7.103190273568361e-01],
        [3.923230223181028e-01, 6.076769776818971e-01],
        [4.999999999999998e-01, 5.000000000000001e-01],
        [6.076769776818971e-01, 3.923230223181028e-01],
        [7.103190273568363e-01, 2.896809726431636e-01],
        [8.031266027349228e-01, 1.968733972650771e-01],
        [8.817598449759076e-01, 1.182401550240923e-01],
        [9.425410221114880e-01, 5.745897788851195e-02],
        [9.826229632519192e-01, 1.737703674808077e-02],
        [1.737703674808060e-02, 0.000000000000000e+00],
        [5.745897788851178e-02, 0.000000000000000e+00],
        [1.182401550240923e-01, 0.000000000000000e+00],
        [1.968733972650772e-01, 0.000000000000000e+00],
        [2.896809726431638e-01, 0.000000000000000e+00],
        [3.923230223181028e-01, 0.000000000000000e+00],
        [4.999999999999998e-01, 0.000000000000000e+00],
        [6.076769776818971e-01, 0.000000000000000e+00],
        [7.103190273568363e-01, 0.000000000000000e+00],
        [8.031266027349228e-01, 0.000000000000000e+00],
        [8.817598449759076e-01, 0.000000000000000e+00],
        [9.425410221114880e-01, 0.000000000000000e+00],
        [9.826229632519192e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.737703674808060e-02],
        [0.000000000000000e+00, 5.745897788851178e-02],
        [0.000000000000000e+00, 1.182401550240923e-01],
        [0.000000000000000e+00, 1.968733972650772e-01],
        [0.000000000000000e+00, 2.896809726431638e-01],
        [0.000000000000000e+00, 3.923230223181028e-01],
        [0.000000000000000e+00, 4.999999999999998e-01],
        [0.000000000000000e+00, 6.076769776818971e-01],
        [0.000000000000000e+00, 7.103190273568363e-01],
        [0.000000000000000e+00, 8.031266027349228e-01],
        [0.000000000000000e+00, 8.817598449759076e-01],
        [0.000000000000000e+00, 9.425410221114880e-01],
        [0.000000000000000e+00, 9.826229632519192e-01],
        ]

_rsv_lebgls[15] = [
        [3.333333333333333e-01, 3.333333333333333e-01],
        [2.259887508865324e-02, 2.259887508865324e-02],
        [6.337752529816408e-02, 6.337752529816408e-02],
        [1.299996108136073e-01, 1.299996108136073e-01],
        [2.259887508865324e-02, 9.548022498226934e-01],
        [6.337752529816408e-02, 8.732449494036718e-01],
        [1.299996108136073e-01, 7.400007783727852e-01],
        [9.548022498226934e-01, 2.259887508865324e-02],
        [8.732449494036718e-01, 6.337752529816408e-02],
        [7.400007783727852e-01, 1.299996108136073e-01],
        [4.768184624021342e-01, 4.768184624021342e-01],
        [4.135242422374686e-01, 4.135242422374686e-01],
        [4.475011342659718e-01, 4.475011342659718e-01],
        [4.768184624021342e-01, 4.636307519573157e-02],
        [4.135242422374686e-01, 1.729515155250627e-01],
        [4.475011342659718e-01, 1.049977314680562e-01],
        [4.636307519573157e-02, 4.768184624021342e-01],
        [1.729515155250627e-01, 4.135242422374686e-01],
        [1.049977314680562e-01, 4.475011342659718e-01],
        [1.984381315028134e-01, 6.660691622793706e-01],
        [2.706592592313675e-01, 6.488056171765984e-01],
        [2.928073839318821e-01, 6.831597592164991e-01],
        [2.628566217358005e-01, 5.392755677725964e-01],
        [3.257909497774679e-01, 5.479384774728655e-01],
        [4.214455528895491e-01, 5.627175182619162e-01],
        [3.583733638754480e-01, 5.880202970479078e-01],
        [1.937972504074645e-01, 7.497461573992948e-01],
        [1.199382949296624e-01, 8.286645968284446e-01],
        [8.498011953666545e-02, 8.970890223173155e-01],
        [3.258604505228824e-01, 4.435185812169222e-01],
        [1.796047914038433e-01, 8.040198765390368e-01],
        [1.984381315028134e-01, 1.354927062178159e-01],
        [2.706592592313675e-01, 8.053512359203396e-02],
        [2.928073839318821e-01, 2.403285685161871e-02],
        [2.628566217358005e-01, 1.978678104916029e-01],
        [3.257909497774679e-01, 1.262705727496664e-01],
        [4.214455528895491e-01, 1.583692884853460e-02],
        [3.583733638754480e-01, 5.360633907664413e-02],
        [1.937972504074645e-01, 5.645659219324061e-02],
        [1.199382949296624e-01, 5.139710824189291e-02],
        [8.498011953666545e-02, 1.793085814601891e-02],
        [3.258604505228824e-01, 2.306209682601952e-01],
        [1.796047914038433e-01, 1.637533205711982e-02],
        [6.660691622793706e-01, 1.984381315028134e-01],
        [6.488056171765984e-01, 2.706592592313675e-01],
        [6.831597592164991e-01, 2.928073839318821e-01],
        [5.392755677725964e-01, 2.628566217358005e-01],
        [5.479384774728655e-01, 3.257909497774679e-01],
        [5.627175182619162e-01, 4.214455528895491e-01],
        [5.880202970479078e-01, 3.583733638754480e-01],
        [7.497461573992948e-01, 1.937972504074645e-01],
        [8.286645968284446e-01, 1.199382949296624e-01],
        [8.970890223173155e-01, 8.498011953666545e-02],
        [4.435185812169222e-01, 3.258604505228824e-01],
        [8.040198765390368e-01, 1.796047914038433e-01],
        [6.660691622793706e-01, 1.354927062178159e-01],
        [6.488056171765984e-01, 8.053512359203396e-02],
        [6.831597592164991e-01, 2.403285685161871e-02],
        [5.392755677725964e-01, 1.978678104916029e-01],
        [5.479384774728655e-01, 1.262705727496664e-01],
        [5.627175182619162e-01, 1.583692884853460e-02],
        [5.880202970479078e-01, 5.360633907664413e-02],
        [7.497461573992948e-01, 5.645659219324061e-02],
        [8.286645968284446e-01, 5.139710824189291e-02],
        [8.970890223173155e-01, 1.793085814601891e-02],
        [4.435185812169222e-01, 2.306209682601952e-01],
        [8.040198765390368e-01, 1.637533205711982e-02],
        [1.354927062178159e-01, 1.984381315028134e-01],
        [8.053512359203396e-02, 2.706592592313675e-01],
        [2.403285685161871e-02, 2.928073839318821e-01],
        [1.978678104916029e-01, 2.628566217358005e-01],
        [1.262705727496664e-01, 3.257909497774679e-01],
        [1.583692884853460e-02, 4.214455528895491e-01],
        [5.360633907664413e-02, 3.583733638754480e-01],
        [5.645659219324061e-02, 1.937972504074645e-01],
        [5.139710824189291e-02, 1.199382949296624e-01],
        [1.793085814601891e-02, 8.498011953666545e-02],
        [2.306209682601952e-01, 3.258604505228824e-01],
        [1.637533205711982e-02, 1.796047914038433e-01],
        [1.354927062178159e-01, 6.660691622793706e-01],
        [8.053512359203396e-02, 6.488056171765984e-01],
        [2.403285685161871e-02, 6.831597592164991e-01],
        [1.978678104916029e-01, 5.392755677725964e-01],
        [1.262705727496664e-01, 5.479384774728655e-01],
        [1.583692884853460e-02, 5.627175182619162e-01],
        [5.360633907664413e-02, 5.880202970479078e-01],
        [5.645659219324061e-02, 7.497461573992948e-01],
        [5.139710824189291e-02, 8.286645968284446e-01],
        [1.793085814601891e-02, 8.970890223173155e-01],
        [2.306209682601952e-01, 4.435185812169222e-01],
        [1.637533205711982e-02, 8.040198765390368e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.521597686489117e-02, 9.847840231351088e-01],
        [5.039973345326398e-02, 9.496002665467360e-01],
        [1.039958540690923e-01, 8.960041459309076e-01],
        [1.738056485587534e-01, 8.261943514412466e-01],
        [2.569702890564312e-01, 7.430297109435687e-01],
        [3.500847655496184e-01, 6.499152344503815e-01],
        [4.493368632390252e-01, 5.506631367609747e-01],
        [5.506631367609745e-01, 4.493368632390254e-01],
        [6.499152344503815e-01, 3.500847655496184e-01],
        [7.430297109435688e-01, 2.569702890564311e-01],
        [8.261943514412466e-01, 1.738056485587533e-01],
        [8.960041459309076e-01, 1.039958540690923e-01],
        [9.496002665467360e-01, 5.039973345326398e-02],
        [9.847840231351090e-01, 1.521597686489095e-02],
        [1.521597686489117e-02, 0.000000000000000e+00],
        [5.039973345326398e-02, 0.000000000000000e+00],
        [1.039958540690923e-01, 0.000000000000000e+00],
        [1.738056485587534e-01, 0.000000000000000e+00],
        [2.569702890564312e-01, 0.000000000000000e+00],
        [3.500847655496184e-01, 0.000000000000000e+00],
        [4.493368632390252e-01, 0.000000000000000e+00],
        [5.506631367609745e-01, 0.000000000000000e+00],
        [6.499152344503815e-01, 0.000000000000000e+00],
        [7.430297109435688e-01, 0.000000000000000e+00],
        [8.261943514412466e-01, 0.000000000000000e+00],
        [8.960041459309076e-01, 0.000000000000000e+00],
        [9.496002665467360e-01, 0.000000000000000e+00],
        [9.847840231351090e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.521597686489117e-02],
        [0.000000000000000e+00, 5.039973345326398e-02],
        [0.000000000000000e+00, 1.039958540690923e-01],
        [0.000000000000000e+00, 1.738056485587534e-01],
        [0.000000000000000e+00, 2.569702890564312e-01],
        [0.000000000000000e+00, 3.500847655496184e-01],
        [0.000000000000000e+00, 4.493368632390252e-01],
        [0.000000000000000e+00, 5.506631367609745e-01],
        [0.000000000000000e+00, 6.499152344503815e-01],
        [0.000000000000000e+00, 7.430297109435688e-01],
        [0.000000000000000e+00, 8.261943514412466e-01],
        [0.000000000000000e+00, 8.960041459309076e-01],
        [0.000000000000000e+00, 9.496002665467360e-01],
        [0.000000000000000e+00, 9.847840231351090e-01],
        ]

_rsv_lebgls[16] = [
        [5.284047111616667e-02, 5.284047111616667e-02],
        [2.645455243559301e-01, 2.645455243559301e-01],
        [2.559645338426510e-02, 2.559645338426510e-02],
        [1.499953744415702e-01, 1.499953744415702e-01],
        [5.284047111616667e-02, 8.943190577676666e-01],
        [2.645455243559301e-01, 4.709089512881395e-01],
        [2.559645338426510e-02, 9.488070932314697e-01],
        [1.499953744415702e-01, 7.000092511168594e-01],
        [8.943190577676666e-01, 5.284047111616667e-02],
        [4.709089512881395e-01, 2.645455243559301e-01],
        [9.488070932314697e-01, 2.559645338426510e-02],
        [7.000092511168594e-01, 1.499953744415702e-01],
        [4.283192720111305e-01, 4.283192720111305e-01],
        [4.772486744600083e-01, 4.772486744600083e-01],
        [3.590252375678640e-01, 3.590252375678640e-01],
        [4.283192720111305e-01, 1.433614559777389e-01],
        [4.772486744600083e-01, 4.550265107998330e-02],
        [3.590252375678640e-01, 2.819495248642718e-01],
        [1.433614559777389e-01, 4.283192720111305e-01],
        [4.550265107998330e-02, 4.772486744600083e-01],
        [2.819495248642718e-01, 3.590252375678640e-01],
        [4.064266340483982e-01, 5.139836528036283e-01],
        [4.408916140714010e-01, 5.391596382131452e-01],
        [3.155166619585914e-01, 5.566613706105387e-01],
        [3.329777549008310e-01, 6.152305573515951e-01],
        [7.665376588027764e-02, 9.117800275056731e-01],
        [2.338026098230794e-01, 5.863699513874213e-01],
        [2.147217887957071e-01, 7.742669553673616e-01],
        [3.367659421500994e-01, 4.592838884885920e-01],
        [3.119742262865882e-01, 6.662160116048213e-01],
        [2.225573376197091e-01, 7.378559340872877e-01],
        [1.042596010644970e-01, 8.219786509556015e-01],
        [1.553621225506644e-01, 7.636963728963229e-01],
        [1.378422477974028e-01, 8.352850255545913e-01],
        [2.315357275962031e-01, 6.738457052373849e-01],
        [4.064266340483982e-01, 7.958971314797347e-02],
        [4.408916140714010e-01, 1.994874771545374e-02],
        [3.155166619585914e-01, 1.278219674308698e-01],
        [3.329777549008310e-01, 5.179168774757381e-02],
        [7.665376588027764e-02, 1.156620661404927e-02],
        [2.338026098230794e-01, 1.798274387894992e-01],
        [2.147217887957071e-01, 1.101125583693118e-02],
        [3.367659421500994e-01, 2.039501693613084e-01],
        [3.119742262865882e-01, 2.180976210859031e-02],
        [2.225573376197091e-01, 3.958672829300302e-02],
        [1.042596010644970e-01, 7.376174797990142e-02],
        [1.553621225506644e-01, 8.094150455301252e-02],
        [1.378422477974028e-01, 2.687272664800577e-02],
        [2.315357275962031e-01, 9.461856716641181e-02],
        [5.139836528036283e-01, 4.064266340483982e-01],
        [5.391596382131452e-01, 4.408916140714010e-01],
        [5.566613706105387e-01, 3.155166619585914e-01],
        [6.152305573515951e-01, 3.329777549008310e-01],
        [9.117800275056731e-01, 7.665376588027764e-02],
        [5.863699513874213e-01, 2.338026098230794e-01],
        [7.742669553673616e-01, 2.147217887957071e-01],
        [4.592838884885920e-01, 3.367659421500994e-01],
        [6.662160116048213e-01, 3.119742262865882e-01],
        [7.378559340872877e-01, 2.225573376197091e-01],
        [8.219786509556015e-01, 1.042596010644970e-01],
        [7.636963728963229e-01, 1.553621225506644e-01],
        [8.352850255545913e-01, 1.378422477974028e-01],
        [6.738457052373849e-01, 2.315357275962031e-01],
        [5.139836528036283e-01, 7.958971314797347e-02],
        [5.391596382131452e-01, 1.994874771545374e-02],
        [5.566613706105387e-01, 1.278219674308698e-01],
        [6.152305573515951e-01, 5.179168774757381e-02],
        [9.117800275056731e-01, 1.156620661404927e-02],
        [5.863699513874213e-01, 1.798274387894992e-01],
        [7.742669553673616e-01, 1.101125583693118e-02],
        [4.592838884885920e-01, 2.039501693613084e-01],
        [6.662160116048213e-01, 2.180976210859031e-02],
        [7.378559340872877e-01, 3.958672829300302e-02],
        [8.219786509556015e-01, 7.376174797990142e-02],
        [7.636963728963229e-01, 8.094150455301252e-02],
        [8.352850255545913e-01, 2.687272664800577e-02],
        [6.738457052373849e-01, 9.461856716641181e-02],
        [7.958971314797347e-02, 4.064266340483982e-01],
        [1.994874771545374e-02, 4.408916140714010e-01],
        [1.278219674308698e-01, 3.155166619585914e-01],
        [5.179168774757381e-02, 3.329777549008310e-01],
        [1.156620661404927e-02, 7.665376588027764e-02],
        [1.798274387894992e-01, 2.338026098230794e-01],
        [1.101125583693118e-02, 2.147217887957071e-01],
        [2.039501693613084e-01, 3.367659421500994e-01],
        [2.180976210859031e-02, 3.119742262865882e-01],
        [3.958672829300302e-02, 2.225573376197091e-01],
        [7.376174797990142e-02, 1.042596010644970e-01],
        [8.094150455301252e-02, 1.553621225506644e-01],
        [2.687272664800577e-02, 1.378422477974028e-01],
        [9.461856716641181e-02, 2.315357275962031e-01],
        [7.958971314797347e-02, 5.139836528036283e-01],
        [1.994874771545374e-02, 5.391596382131452e-01],
        [1.278219674308698e-01, 5.566613706105387e-01],
        [5.179168774757381e-02, 6.152305573515951e-01],
        [1.156620661404927e-02, 9.117800275056731e-01],
        [1.798274387894992e-01, 5.863699513874213e-01],
        [1.101125583693118e-02, 7.742669553673616e-01],
        [2.039501693613084e-01, 4.592838884885920e-01],
        [2.180976210859031e-02, 6.662160116048213e-01],
        [3.958672829300302e-02, 7.378559340872877e-01],
        [7.376174797990142e-02, 8.219786509556015e-01],
        [8.094150455301252e-02, 7.636963728963229e-01],
        [2.687272664800577e-02, 8.352850255545913e-01],
        [9.461856716641181e-02, 6.738457052373849e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.343391168429097e-02, 9.865660883157090e-01],
        [4.456000204221327e-02, 9.554399979577867e-01],
        [9.215187438911487e-02, 9.078481256108851e-01],
        [1.544855096861576e-01, 8.455144903138422e-01],
        [2.293073003349492e-01, 7.706926996650507e-01],
        [3.139127832172615e-01, 6.860872167827384e-01],
        [4.052440132408411e-01, 5.947559867591587e-01],
        [4.999999999999999e-01, 5.000000000000000e-01],
        [5.947559867591587e-01, 4.052440132408412e-01],
        [6.860872167827384e-01, 3.139127832172615e-01],
        [7.706926996650507e-01, 2.293073003349492e-01],
        [8.455144903138422e-01, 1.544855096861577e-01],
        [9.078481256108852e-01, 9.215187438911476e-02],
        [9.554399979577867e-01, 4.456000204221322e-02],
        [9.865660883157091e-01, 1.343391168429086e-02],
        [1.343391168429097e-02, 0.000000000000000e+00],
        [4.456000204221327e-02, 0.000000000000000e+00],
        [9.215187438911487e-02, 0.000000000000000e+00],
        [1.544855096861576e-01, 0.000000000000000e+00],
        [2.293073003349492e-01, 0.000000000000000e+00],
        [3.139127832172615e-01, 0.000000000000000e+00],
        [4.052440132408411e-01, 0.000000000000000e+00],
        [4.999999999999999e-01, 0.000000000000000e+00],
        [5.947559867591587e-01, 0.000000000000000e+00],
        [6.860872167827384e-01, 0.000000000000000e+00],
        [7.706926996650507e-01, 0.000000000000000e+00],
        [8.455144903138422e-01, 0.000000000000000e+00],
        [9.078481256108852e-01, 0.000000000000000e+00],
        [9.554399979577867e-01, 0.000000000000000e+00],
        [9.865660883157091e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.343391168429097e-02],
        [0.000000000000000e+00, 4.456000204221327e-02],
        [0.000000000000000e+00, 9.215187438911487e-02],
        [0.000000000000000e+00, 1.544855096861576e-01],
        [0.000000000000000e+00, 2.293073003349492e-01],
        [0.000000000000000e+00, 3.139127832172615e-01],
        [0.000000000000000e+00, 4.052440132408411e-01],
        [0.000000000000000e+00, 4.999999999999999e-01],
        [0.000000000000000e+00, 5.947559867591587e-01],
        [0.000000000000000e+00, 6.860872167827384e-01],
        [0.000000000000000e+00, 7.706926996650507e-01],
        [0.000000000000000e+00, 8.455144903138422e-01],
        [0.000000000000000e+00, 9.078481256108852e-01],
        [0.000000000000000e+00, 9.554399979577867e-01],
        [0.000000000000000e+00, 9.865660883157091e-01],
        ]

_rsv_lebgls[17] = [
        [1.945791515169107e-01, 1.945791515169107e-01],
        [2.049112087429205e-02, 2.049112087429205e-02],
        [4.259795575418456e-02, 4.259795575418456e-02],
        [1.564181521041115e-01, 1.564181521041115e-01],
        [1.945791515169107e-01, 6.108416969661784e-01],
        [2.049112087429205e-02, 9.590177582514158e-01],
        [4.259795575418456e-02, 9.148040884916308e-01],
        [1.564181521041115e-01, 6.871636957917768e-01],
        [6.108416969661784e-01, 1.945791515169107e-01],
        [9.590177582514158e-01, 2.049112087429205e-02],
        [9.148040884916308e-01, 4.259795575418456e-02],
        [6.871636957917768e-01, 1.564181521041115e-01],
        [3.592196968697690e-01, 3.592196968697690e-01],
        [3.899674085227521e-01, 3.899674085227521e-01],
        [4.556998336553927e-01, 4.556998336553927e-01],
        [4.779584328346175e-01, 4.779584328346175e-01],
        [3.592196968697690e-01, 2.815606062604619e-01],
        [3.899674085227521e-01, 2.200651829544957e-01],
        [4.556998336553927e-01, 8.860033268921452e-02],
        [4.779584328346175e-01, 4.408313433076493e-02],
        [2.815606062604619e-01, 3.592196968697690e-01],
        [2.200651829544957e-01, 3.899674085227521e-01],
        [8.860033268921452e-02, 4.556998336553927e-01],
        [4.408313433076493e-02, 4.779584328346175e-01],
        [2.064981741524552e-01, 7.765619597341436e-01],
        [3.643269502619441e-01, 5.977234247390156e-01],
        [1.356289190443467e-01, 7.886565198452635e-01],
        [1.989384047369088e-01, 6.984622374624075e-01],
        [2.566042921861959e-01, 5.741340329630012e-01],
        [1.364658140077919e-01, 8.412149343742766e-01],
        [8.644084936146931e-02, 8.566967730929346e-01],
        [2.541088096408400e-01, 6.323144327340671e-01],
        [2.765015607981030e-01, 6.704156948701927e-01],
        [3.705134447642415e-01, 4.826648284887848e-01],
        [3.038781746832655e-01, 6.837355849980972e-01],
        [3.625423658269765e-01, 5.525551268328993e-01],
        [4.292890810760650e-01, 5.586509174935030e-01],
        [2.918435801264757e-01, 4.818709044312088e-01],
        [6.803793687523349e-02, 9.226625456558029e-01],
        [1.935420559772826e-01, 7.663263726878712e-01],
        [2.064981741524552e-01, 1.693986611340114e-02],
        [3.643269502619441e-01, 3.794962499904019e-02],
        [1.356289190443467e-01, 7.571456111038976e-02],
        [1.989384047369088e-01, 1.025993578006836e-01],
        [2.566042921861959e-01, 1.692616748508027e-01],
        [1.364658140077919e-01, 2.231925161793135e-02],
        [8.644084936146931e-02, 5.686237754559597e-02],
        [2.541088096408400e-01, 1.135767576250926e-01],
        [2.765015607981030e-01, 5.308274433170423e-02],
        [3.705134447642415e-01, 1.468217267469735e-01],
        [3.038781746832655e-01, 1.238624031863722e-02],
        [3.625423658269765e-01, 8.490250734012416e-02],
        [4.292890810760650e-01, 1.206000143043195e-02],
        [2.918435801264757e-01, 2.262855154423154e-01],
        [6.803793687523349e-02, 9.299517468963514e-03],
        [1.935420559772826e-01, 4.013157133484601e-02],
        [7.765619597341436e-01, 2.064981741524552e-01],
        [5.977234247390156e-01, 3.643269502619441e-01],
        [7.886565198452635e-01, 1.356289190443467e-01],
        [6.984622374624075e-01, 1.989384047369088e-01],
        [5.741340329630012e-01, 2.566042921861959e-01],
        [8.412149343742766e-01, 1.364658140077919e-01],
        [8.566967730929346e-01, 8.644084936146931e-02],
        [6.323144327340671e-01, 2.541088096408400e-01],
        [6.704156948701927e-01, 2.765015607981030e-01],
        [4.826648284887848e-01, 3.705134447642415e-01],
        [6.837355849980972e-01, 3.038781746832655e-01],
        [5.525551268328993e-01, 3.625423658269765e-01],
        [5.586509174935030e-01, 4.292890810760650e-01],
        [4.818709044312088e-01, 2.918435801264757e-01],
        [9.226625456558029e-01, 6.803793687523349e-02],
        [7.663263726878712e-01, 1.935420559772826e-01],
        [7.765619597341436e-01, 1.693986611340114e-02],
        [5.977234247390156e-01, 3.794962499904019e-02],
        [7.886565198452635e-01, 7.571456111038976e-02],
        [6.984622374624075e-01, 1.025993578006836e-01],
        [5.741340329630012e-01, 1.692616748508027e-01],
        [8.412149343742766e-01, 2.231925161793135e-02],
        [8.566967730929346e-01, 5.686237754559597e-02],
        [6.323144327340671e-01, 1.135767576250926e-01],
        [6.704156948701927e-01, 5.308274433170423e-02],
        [4.826648284887848e-01, 1.468217267469735e-01],
        [6.837355849980972e-01, 1.238624031863722e-02],
        [5.525551268328993e-01, 8.490250734012416e-02],
        [5.586509174935030e-01, 1.206000143043195e-02],
        [4.818709044312088e-01, 2.262855154423154e-01],
        [9.226625456558029e-01, 9.299517468963514e-03],
        [7.663263726878712e-01, 4.013157133484601e-02],
        [1.693986611340114e-02, 2.064981741524552e-01],
        [3.794962499904019e-02, 3.643269502619441e-01],
        [7.571456111038976e-02, 1.356289190443467e-01],
        [1.025993578006836e-01, 1.989384047369088e-01],
        [1.692616748508027e-01, 2.566042921861959e-01],
        [2.231925161793135e-02, 1.364658140077919e-01],
        [5.686237754559597e-02, 8.644084936146931e-02],
        [1.135767576250926e-01, 2.541088096408400e-01],
        [5.308274433170423e-02, 2.765015607981030e-01],
        [1.468217267469735e-01, 3.705134447642415e-01],
        [1.238624031863722e-02, 3.038781746832655e-01],
        [8.490250734012416e-02, 3.625423658269765e-01],
        [1.206000143043195e-02, 4.292890810760650e-01],
        [2.262855154423154e-01, 2.918435801264757e-01],
        [9.299517468963514e-03, 6.803793687523349e-02],
        [4.013157133484601e-02, 1.935420559772826e-01],
        [1.693986611340114e-02, 7.765619597341436e-01],
        [3.794962499904019e-02, 5.977234247390156e-01],
        [7.571456111038976e-02, 7.886565198452635e-01],
        [1.025993578006836e-01, 6.984622374624075e-01],
        [1.692616748508027e-01, 5.741340329630012e-01],
        [2.231925161793135e-02, 8.412149343742766e-01],
        [5.686237754559597e-02, 8.566967730929346e-01],
        [1.135767576250926e-01, 6.323144327340671e-01],
        [5.308274433170423e-02, 6.704156948701927e-01],
        [1.468217267469735e-01, 4.826648284887848e-01],
        [1.238624031863722e-02, 6.837355849980972e-01],
        [8.490250734012416e-02, 5.525551268328993e-01],
        [1.206000143043195e-02, 5.586509174935030e-01],
        [2.262855154423154e-01, 4.818709044312088e-01],
        [9.299517468963514e-03, 9.226625456558029e-01],
        [4.013157133484601e-02, 7.663263726878712e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.194722129390074e-02, 9.880527787060993e-01],
        [3.967540732623298e-02, 9.603245926737670e-01],
        [8.220323239095506e-02, 9.177967676090449e-01],
        [1.381603353583786e-01, 8.618396646416213e-01],
        [2.057475828406691e-01, 7.942524171593308e-01],
        [2.827924815439379e-01, 7.172075184560620e-01],
        [3.668186735608596e-01, 6.331813264391403e-01],
        [4.551254532576739e-01, 5.448745467423260e-01],
        [5.448745467423260e-01, 4.551254532576739e-01],
        [6.331813264391406e-01, 3.668186735608593e-01],
        [7.172075184560620e-01, 2.827924815439379e-01],
        [7.942524171593308e-01, 2.057475828406691e-01],
        [8.618396646416213e-01, 1.381603353583786e-01],
        [9.177967676090450e-01, 8.220323239095495e-02],
        [9.603245926737669e-01, 3.967540732623309e-02],
        [9.880527787060993e-01, 1.194722129390068e-02],
        [1.194722129390074e-02, 0.000000000000000e+00],
        [3.967540732623298e-02, 0.000000000000000e+00],
        [8.220323239095506e-02, 0.000000000000000e+00],
        [1.381603353583786e-01, 0.000000000000000e+00],
        [2.057475828406691e-01, 0.000000000000000e+00],
        [2.827924815439379e-01, 0.000000000000000e+00],
        [3.668186735608596e-01, 0.000000000000000e+00],
        [4.551254532576739e-01, 0.000000000000000e+00],
        [5.448745467423260e-01, 0.000000000000000e+00],
        [6.331813264391406e-01, 0.000000000000000e+00],
        [7.172075184560620e-01, 0.000000000000000e+00],
        [7.942524171593308e-01, 0.000000000000000e+00],
        [8.618396646416213e-01, 0.000000000000000e+00],
        [9.177967676090450e-01, 0.000000000000000e+00],
        [9.603245926737669e-01, 0.000000000000000e+00],
        [9.880527787060993e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.194722129390074e-02],
        [0.000000000000000e+00, 3.967540732623298e-02],
        [0.000000000000000e+00, 8.220323239095506e-02],
        [0.000000000000000e+00, 1.381603353583786e-01],
        [0.000000000000000e+00, 2.057475828406691e-01],
        [0.000000000000000e+00, 2.827924815439379e-01],
        [0.000000000000000e+00, 3.668186735608596e-01],
        [0.000000000000000e+00, 4.551254532576739e-01],
        [0.000000000000000e+00, 5.448745467423260e-01],
        [0.000000000000000e+00, 6.331813264391406e-01],
        [0.000000000000000e+00, 7.172075184560620e-01],
        [0.000000000000000e+00, 7.942524171593308e-01],
        [0.000000000000000e+00, 8.618396646416213e-01],
        [0.000000000000000e+00, 9.177967676090450e-01],
        [0.000000000000000e+00, 9.603245926737669e-01],
        [0.000000000000000e+00, 9.880527787060993e-01],
        ]

_rsv_lebgls[18] = [
        [3.333333333333333e-01, 3.333333333333333e-01],
        [1.867832911591294e-01, 1.867832911591294e-01],
        [1.843625980360632e-02, 1.843625980360632e-02],
        [7.974745165252009e-02, 7.974745165252009e-02],
        [4.582737591105361e-02, 4.582737591105361e-02],
        [1.867832911591294e-01, 6.264334176817411e-01],
        [1.843625980360632e-02, 9.631274803927872e-01],
        [7.974745165252009e-02, 8.405050966949597e-01],
        [4.582737591105361e-02, 9.083452481778927e-01],
        [6.264334176817411e-01, 1.867832911591294e-01],
        [9.631274803927872e-01, 1.843625980360632e-02],
        [8.405050966949597e-01, 7.974745165252009e-02],
        [9.083452481778927e-01, 4.582737591105361e-02],
        [4.610274012108529e-01, 4.610274012108529e-01],
        [4.031134167385391e-01, 4.031134167385391e-01],
        [4.393412546532561e-01, 4.393412546532561e-01],
        [4.610274012108529e-01, 7.794519757829410e-02],
        [4.031134167385391e-01, 1.937731665229216e-01],
        [4.393412546532561e-01, 1.213174906934877e-01],
        [7.794519757829410e-02, 4.610274012108529e-01],
        [1.937731665229216e-01, 4.031134167385391e-01],
        [1.213174906934877e-01, 4.393412546532561e-01],
        [2.697535482222230e-01, 5.131123254361209e-01],
        [1.013663979040955e-01, 8.552745279773800e-01],
        [1.577802125320718e-01, 8.143102779121228e-01],
        [5.704260843925303e-02, 9.307086014792093e-01],
        [3.518333725591127e-01, 5.516132715650402e-01],
        [2.643961982569245e-01, 6.635759045983397e-01],
        [1.323082356077286e-01, 7.771019226176585e-01],
        [4.389280715061170e-01, 5.474017498960880e-01],
        [1.756721652834479e-01, 7.043665532533164e-01],
        [3.285410056449763e-01, 6.538843552798158e-01],
        [3.384986151918548e-01, 5.183440648484733e-01],
        [1.066901027121254e-01, 8.801628839826043e-01],
        [4.388925265213433e-01, 5.152283097309622e-01],
        [1.991527726480603e-01, 7.356822819499346e-01],
        [2.703436617821667e-01, 6.001860955307918e-01],
        [3.526900037620688e-01, 6.037477082466996e-01],
        [2.177546267144484e-01, 7.742157293365573e-01],
        [2.463925035152266e-01, 7.226718910081446e-01],
        [3.242851625161429e-01, 4.292054611356153e-01],
        [2.697535482222230e-01, 2.171341263416559e-01],
        [1.013663979040955e-01, 4.335907411852446e-02],
        [1.577802125320718e-01, 2.790950955580528e-02],
        [5.704260843925303e-02, 1.224879008153767e-02],
        [3.518333725591127e-01, 9.655335587584701e-02],
        [2.643961982569245e-01, 7.202789714473578e-02],
        [1.323082356077286e-01, 9.058984177461271e-02],
        [4.389280715061170e-01, 1.367017859779495e-02],
        [1.756721652834479e-01, 1.199612814632355e-01],
        [3.285410056449763e-01, 1.757463907520784e-02],
        [3.384986151918548e-01, 1.431573199596718e-01],
        [1.066901027121254e-01, 1.314701330527023e-02],
        [4.388925265213433e-01, 4.587916374769451e-02],
        [1.991527726480603e-01, 6.516494540200501e-02],
        [2.703436617821667e-01, 1.294702426870414e-01],
        [3.526900037620688e-01, 4.356228799123151e-02],
        [2.177546267144484e-01, 8.029643948994191e-03],
        [2.463925035152266e-01, 3.093560547662865e-02],
        [3.242851625161429e-01, 2.465093763482417e-01],
        [5.131123254361209e-01, 2.697535482222230e-01],
        [8.552745279773800e-01, 1.013663979040955e-01],
        [8.143102779121228e-01, 1.577802125320718e-01],
        [9.307086014792093e-01, 5.704260843925303e-02],
        [5.516132715650402e-01, 3.518333725591127e-01],
        [6.635759045983397e-01, 2.643961982569245e-01],
        [7.771019226176585e-01, 1.323082356077286e-01],
        [5.474017498960880e-01, 4.389280715061170e-01],
        [7.043665532533164e-01, 1.756721652834479e-01],
        [6.538843552798158e-01, 3.285410056449763e-01],
        [5.183440648484733e-01, 3.384986151918548e-01],
        [8.801628839826043e-01, 1.066901027121254e-01],
        [5.152283097309622e-01, 4.388925265213433e-01],
        [7.356822819499346e-01, 1.991527726480603e-01],
        [6.001860955307918e-01, 2.703436617821667e-01],
        [6.037477082466996e-01, 3.526900037620688e-01],
        [7.742157293365573e-01, 2.177546267144484e-01],
        [7.226718910081446e-01, 2.463925035152266e-01],
        [4.292054611356153e-01, 3.242851625161429e-01],
        [5.131123254361209e-01, 2.171341263416559e-01],
        [8.552745279773800e-01, 4.335907411852446e-02],
        [8.143102779121228e-01, 2.790950955580528e-02],
        [9.307086014792093e-01, 1.224879008153767e-02],
        [5.516132715650402e-01, 9.655335587584701e-02],
        [6.635759045983397e-01, 7.202789714473578e-02],
        [7.771019226176585e-01, 9.058984177461271e-02],
        [5.474017498960880e-01, 1.367017859779495e-02],
        [7.043665532533164e-01, 1.199612814632355e-01],
        [6.538843552798158e-01, 1.757463907520784e-02],
        [5.183440648484733e-01, 1.431573199596718e-01],
        [8.801628839826043e-01, 1.314701330527023e-02],
        [5.152283097309622e-01, 4.587916374769451e-02],
        [7.356822819499346e-01, 6.516494540200501e-02],
        [6.001860955307918e-01, 1.294702426870414e-01],
        [6.037477082466996e-01, 4.356228799123151e-02],
        [7.742157293365573e-01, 8.029643948994191e-03],
        [7.226718910081446e-01, 3.093560547662865e-02],
        [4.292054611356153e-01, 2.465093763482417e-01],
        [2.171341263416559e-01, 2.697535482222230e-01],
        [4.335907411852446e-02, 1.013663979040955e-01],
        [2.790950955580528e-02, 1.577802125320718e-01],
        [1.224879008153767e-02, 5.704260843925303e-02],
        [9.655335587584701e-02, 3.518333725591127e-01],
        [7.202789714473578e-02, 2.643961982569245e-01],
        [9.058984177461271e-02, 1.323082356077286e-01],
        [1.367017859779495e-02, 4.389280715061170e-01],
        [1.199612814632355e-01, 1.756721652834479e-01],
        [1.757463907520784e-02, 3.285410056449763e-01],
        [1.431573199596718e-01, 3.384986151918548e-01],
        [1.314701330527023e-02, 1.066901027121254e-01],
        [4.587916374769451e-02, 4.388925265213433e-01],
        [6.516494540200501e-02, 1.991527726480603e-01],
        [1.294702426870414e-01, 2.703436617821667e-01],
        [4.356228799123151e-02, 3.526900037620688e-01],
        [8.029643948994191e-03, 2.177546267144484e-01],
        [3.093560547662865e-02, 2.463925035152266e-01],
        [2.465093763482417e-01, 3.242851625161429e-01],
        [2.171341263416559e-01, 5.131123254361209e-01],
        [4.335907411852446e-02, 8.552745279773800e-01],
        [2.790950955580528e-02, 8.143102779121228e-01],
        [1.224879008153767e-02, 9.307086014792093e-01],
        [9.655335587584701e-02, 5.516132715650402e-01],
        [7.202789714473578e-02, 6.635759045983397e-01],
        [9.058984177461271e-02, 7.771019226176585e-01],
        [1.367017859779495e-02, 5.474017498960880e-01],
        [1.199612814632355e-01, 7.043665532533164e-01],
        [1.757463907520784e-02, 6.538843552798158e-01],
        [1.431573199596718e-01, 5.183440648484733e-01],
        [1.314701330527023e-02, 8.801628839826043e-01],
        [4.587916374769451e-02, 5.152283097309622e-01],
        [6.516494540200501e-02, 7.356822819499346e-01],
        [1.294702426870414e-01, 6.001860955307918e-01],
        [4.356228799123151e-02, 6.037477082466996e-01],
        [8.029643948994191e-03, 7.742157293365573e-01],
        [3.093560547662865e-02, 7.226718910081446e-01],
        [2.465093763482417e-01, 4.292054611356153e-01],
        [1.000000000000000e+00, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.000000000000000e+00],
        [0.000000000000000e+00, 0.000000000000000e+00],
        [1.069411688896010e-02, 9.893058831110399e-01],
        [3.554923592370679e-02, 9.644507640762931e-01],
        [7.376971110167684e-02, 9.262302888983231e-01],
        [1.242528987236936e-01, 8.757471012763062e-01],
        [1.855459313673898e-01, 8.144540686326101e-01],
        [2.558853571596433e-01, 7.441146428403566e-01],
        [3.332475760877505e-01, 6.667524239122494e-01],
        [4.154069882953593e-01, 5.845930117046407e-01],
        [5.000000000000000e-01, 5.000000000000000e-01],
        [5.845930117046408e-01, 4.154069882953591e-01],
        [6.667524239122492e-01, 3.332475760877507e-01],
        [7.441146428403569e-01, 2.558853571596430e-01],
        [8.144540686326102e-01, 1.855459313673897e-01],
        [8.757471012763063e-01, 1.242528987236936e-01],
        [9.262302888983231e-01, 7.376971110167684e-02],
        [9.644507640762930e-01, 3.554923592370695e-02],
        [9.893058831110401e-01, 1.069411688895982e-02],
        [1.069411688896010e-02, 0.000000000000000e+00],
        [3.554923592370679e-02, 0.000000000000000e+00],
        [7.376971110167684e-02, 0.000000000000000e+00],
        [1.242528987236936e-01, 0.000000000000000e+00],
        [1.855459313673898e-01, 0.000000000000000e+00],
        [2.558853571596433e-01, 0.000000000000000e+00],
        [3.332475760877505e-01, 0.000000000000000e+00],
        [4.154069882953593e-01, 0.000000000000000e+00],
        [5.000000000000000e-01, 0.000000000000000e+00],
        [5.845930117046408e-01, 0.000000000000000e+00],
        [6.667524239122492e-01, 0.000000000000000e+00],
        [7.441146428403569e-01, 0.000000000000000e+00],
        [8.144540686326102e-01, 0.000000000000000e+00],
        [8.757471012763063e-01, 0.000000000000000e+00],
        [9.262302888983231e-01, 0.000000000000000e+00],
        [9.644507640762930e-01, 0.000000000000000e+00],
        [9.893058831110401e-01, 0.000000000000000e+00],
        [0.000000000000000e+00, 1.069411688896010e-02],
        [0.000000000000000e+00, 3.554923592370679e-02],
        [0.000000000000000e+00, 7.376971110167684e-02],
        [0.000000000000000e+00, 1.242528987236936e-01],
        [0.000000000000000e+00, 1.855459313673898e-01],
        [0.000000000000000e+00, 2.558853571596433e-01],
        [0.000000000000000e+00, 3.332475760877505e-01],
        [0.000000000000000e+00, 4.154069882953593e-01],
        [0.000000000000000e+00, 5.000000000000000e-01],
        [0.000000000000000e+00, 5.845930117046408e-01],
        [0.000000000000000e+00, 6.667524239122492e-01],
        [0.000000000000000e+00, 7.441146428403569e-01],
        [0.000000000000000e+00, 8.144540686326102e-01],
        [0.000000000000000e+00, 8.757471012763063e-01],
        [0.000000000000000e+00, 9.262302888983231e-01],
        [0.000000000000000e+00, 9.644507640762930e-01],
        [0.000000000000000e+00, 9.893058831110401e-01],
        ]

# Rapetti, Sommariva, & Vianello Symmetric Lebesgue-minimizing #
# Lobatto-Gauss-Legendre nodes: doi:10.1016/j.cam.2011.11.023  #
# ----------------------------END----------------------------- #


def rapetti_sommariva_vianello(n, domain='unit'):
    '''Create Rapetti-Sommariva-Vianello nodes for the triangle.
    that minimize the Lebesgue constant while remaining symmetric
    and having Lobatto-Gauss-Legendre nodes on the edges.

    Notes:

        The Rapetti-Sommariva-Vianello nodes are implicitly defined
        by minimizing the Lebesgue constanat while maintaining `S_3` symmetry
        and edge nodes that are Lobatto-Gauss-Legendre nodes.

    Args:
        n (int): The polynomial degree, must be `\\leq 18`.
        domain (str, optional): The domain of the simplex.  See
            ":ref:`domains`" for the choices and their formal definitions.

    Returns:
        ndarray: Rapetti-Sommariva-Vianellow nodes as a 2D array with one row
        for each of `\\binom{n+2}{2}` nodes, and 2 columns for coordinates
        (or 3 if ``domain='barycentric'``).

    Example:

        This plot shows the Rapetti-Sommariva-Vianello nodes for `n = 10`,
        which do not have the lattice-like structure of the explicitly
        defined nodes in the module.

        .. plot::
           :include-source: True

           >>> import matplotlib.pyplot as plt
           >>> from numpy import eye
           >>> from recursivenodes.nodes import rapetti_sommariva_vianello
           >>> from recursivenodes.utils import coord_map
           >>> nodes = rapetti_sommariva_vianello(10, domain='equilateral')
           >>> corners = coord_map(eye(3), 'barycentric', 'equilateral')
           >>> plt.plot(corners[[0,1,2,0],0], corners[[0,1,2,0],1])
           [<matplotlib.lines.Line2D object at ...>]
           >>> plt.scatter(nodes[:,0], nodes[:,1])
           <matplotlib.collections.PathCollection object at ...>
           >>> plt.gca().set_aspect('equal')
           >>> plt.title('Rapetti-Sommariva-Vianello Nodes')
           Text(0.5, 1.0, 'Rapetti-Sommariva-Vianello Nodes')
           >>> plt.show()

    References:

        :cite:`RaSV12`

    Warning:

        These nodes cannot be used with
        :func:`recursivenodes.lebesgue.lebesguemax` for `n \\geq 10`, because
        they do not have lattice-like structure.
    '''
    nodes = np.array(_rsv_lebgls[n])
    return coord_map(nodes, 'unit', domain)


def expand_to_boundary(d, n, nodes, ex_nodes, domain='biunit'):
    ''' given degree n nodes for the d-simplex that are logically laid out like
    the equispaced nodes and are only in the interior, construct degree (n + d
    + 1) nodes for the d-simplex which are equal to nodes in the interior and
    ex_nodes on the boundary.  No attempt is made to make the new boundary
    nodes good for interpolation: they are just there to aid in computing
    lebesguemax() in lebesgue.py '''
    nd = n + d + 1
    tuple_to_index = {}
    for (k, i) in enumerate(multiindex_equal(d+1, n)):
        tuple_to_index[i] = k
    ex_nodes = ex_nodes.copy()
    for (k, i) in enumerate(multiindex_equal(d+1, nd)):
        if min(i) > 0:
            ihat = tuple([a - 1 for a in i])
            j = tuple_to_index[ihat]
            ex_nodes[k, :] = nodes[j, :]
    return ex_nodes


def add_nodes_to_parser(parser):
    parser.add_argument('-n', '--degree', type=int, nargs='?', default=7,
                        help=('degree of polynomial space for '
                              'which node set is equisolvent'))
    parser.add_argument('-d', '--dimension', type=int, nargs='?', default=3,
                        help='spatial dimensions to node set simplex')
    parser.add_argument('--nodes', type=str,
                        choices=['recursive',
                                 'warburton',
                                 'blyth_luo_pozrikidis',
                                 'equispaced',
                                 'equispaced_interior'],
                        nargs='?', default='recursive',
                        help='node placement algorithm')
    parser.add_argument('-w', '--w-alpha', type=float, nargs='?',
                        help=('blending parameter for '
                              'Warburton warp & blend method'))
    parser.add_argument('--domain', type=str,
                        choices=['biunit',
                                 'unit',
                                 'barycentric',
                                 'equilateral'],
                        default='biunit', nargs='?',
                        help='output simlex domain')
    parser.add_argument('-f', '--family', type=str,
                        choices=['lgl',
                                 'lgc',
                                 'gl',
                                 'gc',
                                 'equi',
                                 'equi_interior',
                                 'lgg',
                                 'gg'],
                        default='lgl', nargs='?',
                        help=('base 1D node family for Blyth-Pozrikidis, '
                              'Warburton, and recursive nodes'))
    parser.add_argument('-g', '--g-alpha', type=float, nargs='?', default=0.,
                        help=('Gegenbauer alpha exponent (only used with '
                              '--family lgg or --family gg)'))


def nodes_from_args(args, expanded=False):
    d = args.dimension
    n = args.degree
    domain = args.domain
    fam_arg = args.family
    need_expanded = False
    if expanded:
        need_expanded = True
        if (args.nodes == 'equispaced' or
                (args.nodes != 'equispaced_interior'
                 and (args.family == 'equi' or fam_arg[0] == 'l'))):
            need_expanded = False
    if need_expanded:
        if (args.nodes == 'equispaced_interior'
                or args.family == 'equi_interior'):
            ex_fam_arg = 'equi'
        else:
            ex_fam_arg = 'l' + fam_arg
    if (fam_arg == 'lgg' or fam_arg == 'gg'):
        fam_arg = (fam_arg, args.g_alpha)
    family = _decode_family(fam_arg)
    x = coord_map(family[n], 'unit', domain)
    if need_expanded:
        if (ex_fam_arg == 'lgg'):
            ex_fam_arg = (ex_fam_arg, args.g_alpha)
        ex_family = _decode_family(ex_fam_arg)
        ex_x = coord_map(family[n+d+1], 'unit', domain)
    if args.nodes == 'equispaced':
        nodes = equispaced(d, n, domain=domain)
    elif args.nodes == 'equispaced_interior':
        nodes = equispaced_interior(d, n, domain=domain)
        if need_expanded:
            ex_nodes = equispaced(d, n+d+1, domain=domain)
    elif args.nodes == 'blyth_luo_pozrikidis':
        nodes = blyth_luo_pozrikidis(d, n, x, domain=domain)
        if need_expanded:
            ex_nodes = blyth_luo_pozrikidis(d, n+d+1, ex_x, domain=domain)
    elif args.nodes == 'warburton':
        alpha = args.w_alpha
        nodes = warburton(d, n, x, alpha=alpha, domain=domain)
        if need_expanded:
            ex_nodes = warburton(d, n+d+1, ex_x, alpha=alpha, domain=domain)
    elif args.nodes == 'recursive':
        nodes = recursive(d, n, family, domain=domain)
        if need_expanded:
            ex_nodes = recursive(d, n+d+1, ex_family, domain=domain)
    if expanded:
        if need_expanded:
            ex_nodes = expand_to_boundary(d, n, nodes, ex_nodes, domain=domain)
            return (d, n, domain, family, nodes, ex_nodes)
        else:
            return (d, n, domain, family, nodes, nodes)
    return (d, n, domain, family, nodes)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=('Print out a node set '
                                         'implemented in nodes.py'))
    add_nodes_to_parser(parser)
    args = parser.parse_args()
    d, n, domain, family, nodes = nodes_from_args(args)
    print(nodes)
