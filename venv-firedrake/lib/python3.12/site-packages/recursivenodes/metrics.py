'''Additional metrics for evaluating a set of nodes'''

import numpy as np
from recursivenodes.polynomials import (
        proriolkoornwinderdubinervandermonde as pkdv,
        proriolkoornwinderdubinervandermondegrad as pkdvgrad,
        proriolkoornwinderdubinervandermondehess as pkdvhess)
from recursivenodes.utils import coord_map, npolys
from recursivenodes.quadrature import simplexgausslegendre


def mass_matrix_condition(d, n, nodes, domain='biunit'):
    '''Compute the condition number of the mass matrix `M_{ij} =
    \\int_{\\Delta^d} \\varphi_i \\varphi_j\\ dx`, where
    `\\{\\varphi_i\\}` is the set of shape functions associated
    with a set of nodes.

    Args:
        d (int): The spatial dimension
        n (int): The polynomial degree
        nodes (ndarray): The set of polynomial nodes, with shape
            `(\\binom{n+d}{d}, d)` (unless ``domain=barycentric``,
            then with shape `(\\binom{n+d}{d}, d+1))`
        domain (str, optional):  The domain of the nodes. See ":ref:`domains`".

    Returns:
        float: the condition number `\\kappa_2(M)`.
    '''
    nodes = coord_map(nodes, domain, 'biunit')
    V = pkdv(d, n, nodes)
    Vinv = np.linalg.inv(V)
    M = Vinv.T.dot(Vinv)
    k = np.linalg.cond(M)
    return k


def weak_laplacian_condition(d, n, nodes, domain='biunit'):
    '''Compute the condition number of the weak-form Laplacian matrix `K_{ij} =
    \\int_{\\Delta^d} \\nabla \\varphi_i \\cdot \\nabla \\varphi_j\\ dx`, where
    `\\{\\varphi_i\\}` is the set of shape functions associated with a set of
    nodes.

    Args:
        d (int): The spatial dimension
        n (int): The polynomial degree
        nodes (ndarray): The set of polynomial nodes, with shape
            `(\\binom{n+d}{d}, d)` (unless ``domain=barycentric``,
            then with shape `(\\binom{n+d}{d}, d+1)`)
        domain (str, optional):  The domain of the nodes. See ":ref:`domains`".
            The condition number will be computed for the nodes mapped to the
            biunit simplex.

    Returns:
        float: the condition number `\\kappa_2(K)` (with respect to the
        pseudoinverse).
    '''
    nodes = coord_map(nodes, domain, 'biunit')
    V = pkdv(d, n, nodes)
    Vinv = np.linalg.inv(V)
    q, w = simplexgausslegendre(d, n)
    npoints = q.ravel().shape[0] // d
    w = w.reshape((npoints,))
    q = q.reshape((npoints, d))
    G = pkdvgrad(d, n, q, C=Vinv)
    K = np.einsum('kil,k,kjl->ij', G, w, G)
    s = np.linalg.svd(K, compute_uv=False)
    k = s[0] / s[-2]
    return k


def strong_laplacian_condition(d, n, nodes, domain='biunit'):
    '''Compute the condition number of the strong-form Laplacian matrix `L_{ij}
    = \\int_{\\Delta^d} \\varphi_i \\cdot \\Delta \\varphi_j\\ dx`, where
    `\\{\\varphi_i\\}` is the set of shape functions associated with a set of
    nodes.

    Args:
        d (int): The spatial dimension
        n (int): The polynomial degree
        nodes (ndarray): The set of polynomial nodes, with shape
            `(\\binom{n+d}{d}, d)` (unless ``domain=barycentric``,
            then with shape `(\\binom{n+d}{d}, d+1))`
        domain (str, optional):  The domain of the nodes. See ":ref:`domains`".
            The condition number will be computed for the nodes mapped to the
            biunit simplex.

    Returns:
        float: the condition number `\\kappa_2(L)` (with respect to
        the pseudoinverse).
    '''
    nodes = coord_map(nodes, domain, 'biunit')
    V = pkdv(d, n, nodes)
    Vinv = np.linalg.inv(V)
    q, w = simplexgausslegendre(d, n)
    npoints = q.ravel().shape[0] // d
    w = w.reshape((npoints,))
    if n >= 2:
        nharm = npolys(d, n) - npolys(d, n-2)
    else:
        return 0.
    q = q.reshape((npoints, d))
    B = pkdv(d, n, q, C=Vinv)
    H = pkdvhess(d, n, q, C=Vinv)
    L = np.einsum('ki,k,kjll->ij', B, w, H)
    s = np.linalg.svd(L, compute_uv=False)
    k = s[0] / s[-(nharm+1)]
    return k


def nodal_laplacian_condition(d, n, nodes, domain='biunit'):
    '''Compute the condition number of the nodal strong-form Laplacian matrix
    `L_{ij} = \\Delta \\varphi_j(x_i)`, where `\\{\\varphi_i\\}` is the
    set of shape functions associated with a set of nodes.

    Args:
        d (int): The spatial dimension
        n (int): The polynomial degree
        nodes (ndarray): The set of polynomial nodes, with shape
            `(\\binom{n+d}{d}, d)` (unless ``domain=barycentric``,
            then with shape `(\\binom{n+d}{d}, d+1))`
        domain (str, optional):  The domain of the nodes. See ":ref:`domains`".
            The condition number will be computed for the nodes mapped to the
            biunit simplex.

    Returns:
        float: the condition number `\\kappa_2(L)` (with respect
        to the pseudoinverse).
    '''
    nodes = coord_map(nodes, domain, 'biunit')
    V = pkdv(d, n, nodes)
    Vinv = np.linalg.inv(V)
    if n >= 2:
        nharm = npolys(d, n) - npolys(d, n-2)
    else:
        return 0.
    H = pkdvhess(d, n, nodes, C=Vinv)
    L = np.einsum('ijll->ij', H)
    s = np.linalg.svd(L, compute_uv=False)
    k = s[0] / s[-(nharm+1)]
    return k


def nodal_gradient_condition(d, n, nodes, domain='biunit'):
    '''Compute the condition number of the nodal gradient matrix
    `G_{(d i + k)j} = \\nabla_k \\varphi_j(x_i)`, where `\\{\\varphi_i\\}`
    is the set of shape functions associated with a set of nodes.

    Args:
        d (int): The spatial dimension
        n (int): The polynomial degree
        nodes (ndarray): The set of polynomial nodes, with shape
            `(\\binom{n+d}{d}, d)` (unless ``domain=barycentric``,
            then with shape `(\\binom{n+d}{d}, d+1))`
        domain (str, optional):  The domain of the nodes. See ":ref:`domains`".
            The condition number will be computed for the nodes mapped to the
            biunit simplex.

    Returns:
        float: the condition number `\\kappa_2(G)` (with respect
        to the pseudoinverse).
    '''
    nodes = coord_map(nodes, domain, 'biunit')
    V = pkdv(d, n, nodes)
    Vinv = np.linalg.inv(V)
    npoints = nodes.shape[0]
    G = pkdvgrad(d, n, nodes, C=Vinv)
    G = np.swapaxes(G, 1, 2).reshape((npoints * d, npoints))
    s = np.linalg.svd(G, compute_uv=False)
    k = s[0] / s[-2]
    return k
