
import numpy as np
from scipy.sparse import block_diag
from recursivenodes.polynomials import (
        proriolkoornwinderdubinervandermonde,
        proriolkoornwinderdubinervandermondegrad,
        proriolkoornwinderdubinervandermondehess)
from recursivenodes.utils import multiindex_up_to, multiindex_equal


def lebesgue(x, d, n, nodes, Vinv=None):
    '''The Lebesgue function in the biunit `d`-simplex with a choice of nodes
    for degree-`n` polynomials.

    Notes:

        Let `X_n = \\{\\boldsymbol{x}_i\\}` be the set of `\\binom{n+d}{d}` nodes.
        The associated shape functions `\\{\\phi_i\\}` are polynomials with
        degree at most `n` such that `\\phi_i(\\boldsymbol{x}_j) =
        \\delta_{ij}`.

        The Lebesgue function is the sum of the absolute value of the shape
        functions,

        .. math::

            l(\\boldsymbol{x}) =
            \\sum_{i=1}^{\\binom{n+d}{d}} | \\phi_i(\\boldsymbol{x}) |.

    Args:

        x (ndarray): Shape (`N`, `d`), points at which to evaluate the Lebesgue
            function.
        d (int): The spatial dimension.
        n (int): The polynomial degree that the nodes are unisolvent for.
        nodes (ndarray): Shape (`\\binom{n+d}{d}`, `d`), the nodes defining the
            shape functions.
        Vinv (ndarray, optional): Shape (`\\binom{n+d}{d}`, `\\binom{n+d}{d}`),
            the inverse of the Vandermonde matrix for the
            Proriol-Koornwinder-Dubiner basis evaluated at the nodes.  These
            polynomials are used as a basis for stable evluation in the biunit
            `d`-simplex, so the inverse of the Vandermonde matrix must be
            computed if it is not already known.

    Returns:
        ndarray: Shape (`N`,), the value of the Lebesgue function for the nodes
        at the points ``x``.

    See Also:
        :func:`recursivenodes.polynomials.proriolkoornwinderdubinervandermonde`.
    '''
    d = nodes.shape[-1]
    if Vinv is None:
        V = proriolkoornwinderdubinervandermonde(d, n, nodes)
        Vinv = np.linalg.inv(V)
    v = proriolkoornwinderdubinervandermonde(d, n, x, C=Vinv)
    return np.abs(v).sum(axis=-1)


def lebesguegrad(x, d, n, nodes, Vinv=None):
    '''Evaluate the gradient of lebesgue(x, d, n, nodes) w.r.t. x

    x is an (k, d) numpy array for k points in the d-simplex

    nodes is a ((n+d) choose d, d) array of nodes in the d-simplex

    returns a numpy array with the same shape as x

    if the inverse of the Vandermonde matrix of the PKD polynomials evaluated
    at the nodes is already known, it can be passed as Vinv '''
    if Vinv is None:
        V = proriolkoornwinderdubinervandermonde(d, n, nodes)
        Vinv = np.linalg.inv(V)
    v, g = proriolkoornwinderdubinervandermondegrad(d, n, x, C=Vinv, both=True)
    s = np.sign(v)
    return (s.reshape(s.shape + (1,)) * g).sum(axis=-2)


def lebesguemin(x, d, n, nodes, Vinv=None):
    '''`-\\sum_j l(\\boldsymbol{x}_j)` and its gradient for the Lebesgue
    function `l` defined by a set of nodes.

    Args:

        x (ndarray): Shape (`N`, `d`), points at which to evaluate the Lebesgue
            function.
        d (int): The spatial dimension.
        n (int): The polynomial degree that the nodes are unisolvent for.
        nodes (ndarray): Shape (`\\binom{n+d}{d}`, `d`), the nodes defining the
            shape functions.
        Vinv (ndarray, optional): See documentation for :func:`lebesgue`.

    Returns:
        (float, ndarray): `-\\sum_j l(\\boldsymbol{x}_j)` and its gradient with
        respect to ``x``, reshaped to a vector with shape (`Nd`,).  (This
        function and return format are intended for use by
        scipy.optimize.minimize_.)

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy-1.4.1/reference/generated/scipy.optimize.minimize.html
    '''
    npoint = x.shape[0] // d
    x = x.reshape((npoint, d))
    if Vinv is None:
        V = proriolkoornwinderdubinervandermonde(d, n, nodes)
        Vinv = np.linalg.inv(V)
    v, g = proriolkoornwinderdubinervandermondegrad(d, n, x, C=Vinv, both=True)
    s = np.sign(v)
    grad = (-(s.reshape(s.shape + (1,)) * g).sum(axis=-2))
    return (-np.abs(v).sum(), grad.reshape((x.shape[0]*x.shape[1],)))


def lebesgueminhess(x, d, n, nodes, Vinv=None):
    '''The Hessian of `-\\sum_j l(\\boldsymbol{x}_j)`  for the Lebesgue
    function `l` defined by a set of nodes.

    Args:

        x (ndarray): Shape (`N`, `d`), points at which to evaluate the Lebesgue
            function.
        d (int): The spatial dimension.
        n (int): The polynomial degree that the nodes are unisolvent for.
        nodes (ndarray): Shape (`\\binom{n+d}{d}`, `d`), the nodes defining the
            shape functions.
        Vinv (ndarray, optional): See documentation for :func:`lebesgue`.

    Returns:
        scipy.sparse.block_diag_: A block diagonal matrix of shape `(Nd,Nd)`,
        the Hessian of the Lebesgue function with respect to ``x``.  (This
        function and return format are intended for use by
        scipy.optimize.minimize_.)

    .. _scipy.sparse.block_diag: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.block_diag.html
    '''
    npoint = x.shape[0] // d
    x = x.reshape((npoint, d))
    if Vinv is None:
        V = proriolkoornwinderdubinervandermonde(d, n, nodes)
        Vinv = np.linalg.inv(V)
    v = proriolkoornwinderdubinervandermonde(d, n, x, C=Vinv)
    h = proriolkoornwinderdubinervandermondehess(d, n, x, C=Vinv)
    s = np.sign(v)
    lmhess = (-(s.reshape(s.shape + (1, 1)) * h).sum(axis=-3))
    lmhesssp = block_diag(lmhess)
    return lmhesssp


def initial_points(d, n, nodes):
    '''Given a set of symmetric nodes on the biunit d-simplex to which the
    equispaced nodes can be deformed without singularities or orientation
    reversals, compute a minimal set of initial guesses which should be
    sufficient to find all local maxima (up to symmetry) '''
    # enumerate barycentric multiindices that are monotonic: these encompass a
    # subset of the d-simplex whose orbit under vertex permutation covers the
    # whole simplex
    iter_tuples = []
    tuple_to_coord = {}
    for (k, ib) in enumerate(multiindex_equal(d+1, n)):
        tuple_to_coord[ib] = nodes[k, :]
        if all(np.array(ib[0:-1]) <= np.array(ib[1:])):
            iter_tuples.append(ib)
    xinit = []
    # The equispaced nodes induce cells with d types
    # for instance, if d == 1 all cells are just intervals
    #               if d == 2 all cells are either upward or downward triangles
    #               if d == 3 there are upward triangles, octahedra, and
    #               downward triangles
    # each type can be characterize as the centroid of a barycentric stencil of
    # width 1, 2, ..., d we start by enumerating these stencils
    for s in range(1, d+1):
        stencil = []
        for i in multiindex_up_to(d, s):
            ib = i + (s-sum(i),)
            # the cell at the center of the stencil is bounded by multiindices
            # containing only 0s and 1s
            if max(ib) == 1:
                # record the stencil as a barycentric stencil, i.e. one whose
                # sum is 0 so that the barycentric condition isn't violated
                ib = i + (-sum(i),)
                stencil.append(ib)
        for t in iter_tuples:  # for every node location (up to symmetry)
            # compute the (equispaced) centroid of the stencil which has this
            # location as its first vertex, in barycentric coordinates
            mp = np.array([((d+1)*t[i]+s) for i in range(d)]+[(d+1)*t[-1]-s*d])
            # if these barycentric coordinates are non-monotonic, then this
            # stencil centroid is in the orbit of another one
            if any(mp[0:-1] > mp[1:]):
                continue
            point = np.zeros(d)
            # compute the centroid for these node coordinates
            for offset in stencil:
                tupl = tuple((t[j] + offset[j]) for j in range(d+1))
                c = tuple_to_coord[tupl]
                point += c
            point /= len(stencil)
            xinit.append(point)
    xinit = np.array(xinit)
    return (iter_tuples, tuple_to_coord, xinit)


def lebesguemax(d, n, nodes, Vinv=None,
                solver_options={'gtol': 1.e-3, 'barrier_tol': 1.e-3},
                ex_nodes=None, logger=None):
    '''An estimate of the Lebesgue constant `\\Lambda_n^{\\max}(\\Delta^d)`,
    the maximum of the Lebesgue function on the biunit `d`-simplex for a set of
    nodes for polynomials up to degree `n`.

    Notes:

        If `l` is the Lebesgue function defined by the nodes (see
        :func:`lebesgue`), then

        .. math::
            :label: lmax

            \\Lambda_n^{\\max}(\\Delta^d) =
            \\max_{\\boldsymbol{x}\\in\\Delta^d} l(\\boldsymbol{x}).

        This constant is significant in quantifying the interpolation quality
        of a set of nodes because of the error estimate

        .. math::

            \\| f - p^n(f) \\|_{\\infty} \\leq (1 + \\Lambda_n^{\\max})
            \\inf_{q\\in\\mathcal{P}_n(\\Delta^d)} \\| f - q \\|_{\\infty},

        where `p^n(f)` is the interpolant of `f` defined by the nodes.

        `\\Lambda_n^{\\max}(\\Delta^d)` is nonconvex and only piecewise smooth,
        with many local maxima.  This function is designed to estimate
        `\\Lambda_n^{\\max}(\\Delta^d)` when the nodes have the same
        lattice-like layout as equispaced nodes (as all but one of the node
        sets implemented in this package do, see :mod:`nodes`).

        When this lattice-like structure is present, the maxima always occur in
        one of the "void"s between nodes in the lattice:

        .. plot::
            :include-source: True

            >>> import matplotlib.pyplot as plt
            >>> from recursivenodes import recursive_nodes
            >>> from recursivenodes.nodes import equispaced
            >>> from recursivenodes.utils import coord_map
            >>> from recursivenodes.lebesgue import lebesgue
            >>> x = equispaced(2, 200, domain='biunit')
            >>> nodes = recursive_nodes(2, 6, domain='biunit')
            >>> l = lebesgue(x, 2, 6, nodes)
            >>> x = coord_map(x, 'biunit', 'equilateral')
            >>> ax = plt.gca()
            >>> ax.tricontour(x[:,0], x[:, 1], l, 41)
            <matplotlib.tri.tricontour.TriContourSet object at ...>
            >>> ax.set_aspect('equal')
            >>> plt.show()

        Also, though the Lebesgue function is only piecewise smooth, it is
        guaranteed to be smooth at the maxima.  An efficient way to estimate
        the `\\Lambda_n^{\\max}(\\Delta^d)` for nodes with lattice-like
        structure is to try to optimize the Lebesgue function from one initial
        guess at the centroid of each void.  The node sets in this package are
        also fully symmetric, which means that initial guesses can be placed in
        only a fraction of the voids.  Here, for instance, are the 3D degree 7
        equispaced nodes and the initial guesses used by ``lebesguemax``:

        .. plot::

            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from recursivenodes import recursive_nodes
            from recursivenodes.nodes import equispaced
            from recursivenodes.lebesgue import initial_points
            from recursivenodes.utils import coord_map
            nodes = equispaced(3, 7, domain='biunit')
            _, _, xinit = initial_points(3, 7, nodes)
            nodes = coord_map(nodes, 'biunit', 'equilateral')
            xinit = coord_map(xinit, 'biunit', 'equilateral')
            ax = plt.gca(projection='3d')
            ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2])
            ax.scatter(xinit[:,0], xinit[:,1], xinit[:,2])
            ax.view_init(None, 225)
            ax.axis('off')
            ax.auto_scale_xyz([-0.7, 0.7], [-0.7, 0.7], [-0.55, 0.85])
            ax.dist = 8
            plt.tight_layout()
            plt.show()


        This function uses scipy.optimize.minimize_ (method='trust-constr') to
        find the local maxima from the initial guesses.  While it is not
        theoretically guaranteed that the global maximum is always found in
        this way, the values of `\\Lambda_n^{\\max}(\\Delta^d)` that have been
        compute with ``lebesguemax`` agree with previously computed values in
        :cite:`BlPo06,LuPo06,Warb06`.

    Args:

        d (int): The spatial dimension.
        n (int): The polynomial degree that the nodes are unisolvent for.
        nodes (ndarray): Shape (`\\binom{n+d}{d}`, `d`), the nodes defining the
            shape functions.
        Vinv (ndarray, optional): See documentation for :func:`lebesgue`.
        solver_options (dict, optional): Options to control the behavior of
            scipy.optimize.minimize_.
        ex_nodes (ndarray, optional): if ``nodes`` includes nodes that do not
            touch the boundary of the simplex, ``ex_nodes`` is a set of nodes
            that includes extra boundary locations that can be used to help
            define initial guesses.
        logger (logging.logger, optional): Logger for progress information.

    Returns:
        (float, ndarray): The estimate and estimated argmax of :eq:`lmax`.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy-1.4.1/reference/generated/scipy.optimize.minimize.html
    '''

    if Vinv is None:
        V = proriolkoornwinderdubinervandermonde(d, n, nodes)
        Vinv = np.linalg.inv(V)

    if not (ex_nodes is None) and not (ex_nodes is nodes):
        iter_tuples, tuple_to_coord, xinit = initial_points(d, n+d+1, ex_nodes)
    else:
        iter_tuples, tuple_to_coord, xinit = initial_points(d, n, nodes)
    npoints = xinit.shape[0]

    options = solver_options.copy()
    # use the distance between the first two nodes to estimate the initial
    # trust region area
    xscale = np.linalg.norm(nodes[1, :] - nodes[0, :])
    options['initial_tr_radius'] = xscale / 10.

    from scipy.optimize import minimize, LinearConstraint
    import scipy.sparse
    constraints = None
    # compute the boundaries of the biunit simplex as bounds for optimization
    # problems
    A = np.concatenate((np.eye(d), np.ones((d,)).reshape((1, d))), axis=0)
    lb = -1 * np.ones((d+1,))
    lb[-1] = -d
    ub = np.ones((d+1,))
    ub[-1] = 2-d
    max_x = xinit.copy()
    # group multiple initial points for better BLAS intensity (although this
    # may cause multiple points to iterate as many iterations as the worst in
    # the group needs)
    block = 8
    for i in range(int(np.ceil(npoints/block))):
        prev = i*block
        last = min(npoints, (i+1)*block)
        size = last - prev
        Ablock = scipy.sparse.kron(scipy.sparse.eye(size), A)
        lbblock = np.tile(lb, size)
        ubblock = np.tile(ub, size)
        constraints = LinearConstraint(Ablock, lbblock, ubblock,
                                       keep_feasible=True)
        x = xinit[prev:last, :].reshape((size*d,))
        Ax = Ablock.dot(x)
        assert (Ax >= lbblock).all()
        assert (Ax <= ubblock).all()
        sol = minimize(lebesguemin, x, (d, n, nodes, Vinv), jac=True,
                       hess=lebesgueminhess, method='trust-constr',
                       options=options, constraints=constraints)
        if (sol.message != "`gtol` termination condition is satisfied." and
                sol.message != "`xtol` termination condition is satisfied."):
            import warnings
            warnings.warn(f'scipy.optimize.minimize message: {sol.message}')
            warnings.warn(f'scipy.optimize.minimize function value: {sol.fun}')
            warnings.warn(f'scipy.optimize.minimize solution value: {sol.x}')
        if logger:
            logger.info(f'{last} of {npoints}')
        max_x[prev:last, :] = sol.x.reshape((size, d))
    max_l = lebesgue(max_x, d, n, nodes, Vinv)
    imax = np.argmax(max_l)
    return (max_l[imax], max_x[imax, :])


_lebesguemax_ref = None


def _get_lebesguemax_ref():
    import os
    import csv
    mydir = os.path.dirname(__file__)
    ref = {}
    for filename in ['lebesgue_constants.csv', 'more_lebesgue_constants.csv']:
        datafile = os.path.join(mydir, filename)
        with open(datafile, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if (len(row) > 0 and row[0][0] == '#'):
                    continue
                ref[(row[0], int(row[1]), int(row[2]))] = float(row[3])
    return ref


def lebesguemax_reference(d, n, nodes='recursive'):
    '''Look up previously computed Lebesgue constants
    (`\\Lambda_n^{\\max}(\\Delta^d)`) for node sets implemented in this package
    and other node sets in the literature.

    Args:

        d (int): The spatial dimension.
        n (int): The polynomial degree that the nodes are unisolvent for.
        nodes (str, optional): The name of the node set to query.
            Choices include:

            - ``'recursive'``: :func:`recursivenodes.recursive_nodes`.
            - ``'equispaced'``: :func:`recursivenodes.nodes.equispaced`.
            - ``'warburton'``: :func:`recursivenodes.nodes.warburton`.
            - ``'blyth_luo_pozrikidis'``:
              :func:`recursivenodes.nodes.blyth_luo_pozrikidis`.
            - ``'Chen-Babuska'``: Chen-Babuška type nodes
              :cite:`ChBa95,ChBa96`, computed in :cite:`Roth05`.
            - ``'Fekete'``: Fekete nodes, computed in :cite:`Roth05`.
            - ``'Lebesgue'``: Lebesgue-constant-minimizing nodes, computed in
              :cite:`Roth05`.
            - ``'Heinrichs-asymmetric'``: Lebesgue-constant-minimizing nodes
              :cite:`Hein05`.
            - ``'Heinrichs-symmetric'``: Symmetric Lebesgue-constant-minimizing
              nodes :cite:`Hein05`.
            - ``'Hesthaven'``: Hesthaven electrostatic nodes :cite:`Hest98`.
            - ``'Hesthaven-Teng'``: Hesthaven-Teng electrostatic nodes
              :cite:`Hest98`.
            - ``'Rapetti-Sommariva-Vianello-leb'``:
              Lebesgue-constant-minimizing nodes, computed in :cite:`RaSV12`.
            - ``'Rapetti-Sommariva-Vianello-lebgl'``:
              Lebesgue-constant-minimizing nodes wih Lobatto-Gauss-Legendre
              edges, computed in :cite:`RaSV12`.
            - ``'Rapetti-Sommariva-Vianello-lebgls'``: Symmetric
              Lebesgue-constant-minimizing nodes wih Lobatto-Gauss-Legendre
              edges, computed in :cite:`RaSV12`.

    Returns:
        float: `\\Lambda_n^{\\max}(\\Delta^d)` for that node set.

    Raises:
        KeyError: The node set is unknown, or `\\Lambda_n^{\\max}(\\Delta^d)`
            has not been precomputed for it.
    '''
    global _lebesguemax_ref
    if _lebesguemax_ref is None:
        _lebesguemax_ref = _get_lebesguemax_ref()
    return _lebesguemax_ref[(nodes, d, n)]


def add_lebesguemax_to_parser(parser):
    from recursivenodes.nodes import add_nodes_to_parser
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='show verbose computation progress')
    parser.add_argument('-t', '--tol', type=float, nargs='?', default=1.e-3,
                        help='convergence tolerance for gradient convergence')
    add_nodes_to_parser(parser)


def lebesguemax_from_args(args):
    from recursivenodes.nodes import nodes_from_args
    from recursivenodes.utils import coord_map
    d, n, domain, family, nodes, exnodes = nodes_from_args(args, expanded=True)
    if domain != 'biunit':
        new_nodes = coord_map(nodes, domain, 'biunit')
        if exnodes is nodes:
            new_exnodes = new_nodes
        else:
            new_exnodes = coord_map(exnodes, domain, 'biunit')
        nodes = new_nodes
        exnodes = new_exnodes
    solver_options = {'gtol': args.tol, 'barrier_tol': args.tol}
    logger = None
    if args.verbose:
        import logging
        logging.basicConfig()
        logger = logging.getLogger('lebesguemax')
        logger.setLevel('INFO')
    lmaxreg = lebesguemax(d, n, nodes, ex_nodes=exnodes,
                          solver_options=solver_options, logger=logger)
    return lmaxreg


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=(
        'Compute the Lebesgue constant on the d-simplex for a node set '
        'implemented in nodes.py'))
    add_lebesguemax_to_parser(parser)
    args = parser.parse_args()
    (max_l, max_x) = lebesguemax_from_args(args)
    if args.verbose:
        print((max_l, max_x))
    else:
        print(max_l)
