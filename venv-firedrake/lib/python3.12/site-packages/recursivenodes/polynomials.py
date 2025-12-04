'''Polynomials used in ``recursivenodes``.'''

import numpy as np
from math import lgamma
from recursivenodes.utils import multiindex_up_to, npolys


def _jacobi_recurrence(n, alpha=0., beta=0.):
    '''Three term recurrence coefficients for Jacobi polynomials,

    $$ P_n = (a_n x + b_n) P_{n-1} - c_n P_{n-2}. $$

    returns (a_n, b_n, c_n)
    '''
    n = np.array(n, dtype='int')
    a = np.zeros(n.shape)
    b = np.zeros(n.shape)
    c = np.zeros(n.shape)
    a[n == 1] = (alpha+beta+2)/2
    b[n == 1] = (alpha-beta)/2
    N = n[n > 1]
    a[n > 1] = (2*N+alpha+beta)*(2*N+alpha+beta-1) / (2*N*(N+alpha+beta))
    b[n > 1] = ((alpha*alpha-beta*beta)*(2*N+alpha+beta-1)
                / (2*N*(N+alpha+beta)*(2*N+alpha+beta-2)))
    c[n > 1] = (2*(N+alpha-1)*(N+beta-1)*(2*N+alpha+beta)
                / (2*N*(N+alpha+beta)*(2*N+alpha+beta-2)))
    return (a, b, c)


def _eval_jacobi(n, alpha, beta, x, out):
    '''reimplementation of scipy.special.eval_jacobi (only for integer n)'''
    x = np.array(x)
    if out is None:
        out = np.ones(x.shape)
    else:
        out[:] = 1.
    n = int(n)
    if n == 0:
        return out
    ns = list(range(1, n+1))
    a, b, c = _jacobi_recurrence(ns, alpha, beta)
    pm1 = np.zeros(out.shape)
    pm2 = np.ndarray(out.shape)
    for i in range(n):
        pm2[...] = pm1[...]
        pm1[...] = out[...]
        out[...] = (a[i]*x+b[i])*pm1 - c[i]*pm2
    return out


try:
    from scipy.special import eval_jacobi
except ImportError:
    eval_jacobi = _eval_jacobi


def jacobi(n, x, a=0., b=0., out=None):
    '''Evaluation of a Jacobi polynomial.

    Args:
        n (int): The degree of the polynomial.
        x (ndarray): Points at which to evaluate the polynomial.
        a (float, optional): Left exponent of weight function.
        b (float, optional): Right exponent of weight function.
        out (ndarray, optional): Output placed here if given, same shape as
            ``x``.

    Returns:
        ndarray: same shape as ``x``, values of the `n`-th Jacobi polynomials
        with respect to the weight function `(1+x)^a(1-x)^b` at the points in
        ``x``.
    '''
    return eval_jacobi(n, a, b, x, out)


def jacobider(n, x, a=0., b=0., k=1, out=None):
    '''Evaluation of a the `k`-th derivative of a Jacobi polynomial.

    Args:
        n (int): The degree of the polynomial.
        x (ndarray): Points at which to evaluate the polynomial.
        a (float, optional): Left exponent of weight function.
        b (float, optional): Right exponent of weight function.
        out (ndarray, optional): Output placed here if given, same shape as
            ``x``.

    Returns:
        ndarray: same shape as ``x``, values of the `k`-th derivative of the
        `n`-th Jacobi polynomials with respect to the weight function
        `(1+x)^a(1-x)^b` at the points in ``x``.
    '''
    if k > n:
        if out:
            out[:] = 0.
            return out
        else:
            return np.zeros(np.shape(x))
    return jacobi(n-k, x, a+k, b+k) * np.exp(lgamma(a+b+n+1+k) -
                                             lgamma(a+b+n+1)) / 2**k


def jacobinorm2(n, a=0., b=0.):
    '''The square of the weighted `L_2` norm of a Jacobi polynomial.

    Args:
        n (int): The degree of the polynomial.
        x (ndarray): Points at which to evaluate the polynomial.
        a (float, optional): Left exponent of weight function.
        b (float, optional): Right exponent of weight function.

    Returns:
        float: `\\int_{-1}^1 (1+x)^a (1-x)^b P^{(a,b)}_n(x)^2\\ dx`.
    '''
    if n == 0:
        return 2.**(a+b+1) * np.exp(lgamma(a + 1) + lgamma(b + 1) -
                                    lgamma(a + b + 2))
    else:
        return (2.**(a+b+1) / (2*n + a + b + 1) *
                np.exp(lgamma(n + a + 1) + lgamma(n + b + 1) -
                       lgamma(n + a + b + 1) - lgamma(n + 1)))


def proriolkoornwinderdubiner(d, i, x, out=None):
    '''Evaluation of a Proriol-Koornwinder-Dubiner (PKD) polynomial.

    Args:
        d (int): The spatial dimension.
        i (tuple(int)): Multi-index of length `d`, the degree of the leading
            monomial of the PKD polynomial in each spatial coordinate.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial.
        out (ndarray, optional): Shape (`N`,) array to hold output.

    Returns:
        ndarray: Shape (`N`,), evaluation of the PKD polynomial with leading
        monomial degrees `i` at the points `x`.  The PKD polynomials are
        orthonormal on the biunit simplex.

    References:
        :cite:`Pror57,KaMc64,Koor75,Dubi91`
    '''
    if (d == 1):
        return jacobi(i[0], x[:, 0], out=out) / jacobinorm2(i[0])**0.5
    else:
        isum = sum(i[0:d-1])
        factor = (1. - x[:, d-1])
        tol = 1.e-10
        nonzero = np.abs(factor) > tol
        if out is None:
            px = np.ndarray(x[:, 0:(d-1)].shape)
        else:
            px = out
        px[nonzero, :] = ((x[nonzero, 0:(d-1)] + 1.) * 2. /
                          factor[nonzero, np.newaxis] - 1.)
        px[~nonzero, :] = -1.
        pi = proriolkoornwinderdubiner(d-1, i[0:d-1], px)
        pi *= jacobi(i[-1], x[:, d-1], 2*isum + d - 1, 0.)
        pi /= jacobinorm2(i[-1], 2*isum+d-1, 0.)**0.5
        pi *= factor**isum
        pi *= 2.**((d-1)/2)
        return pi


def proriolkoornwinderdubinergrad(d, i, x, out=None, both=False):
    '''Evaluation of the gradient of a Proriol-Koornwinder-Dubiner (PKD)
    polynomial.

    Args:
        d (int): The spatial dimension.
        i (tuple(int)): Multi-index of length `d`, the degree of the leading
            monomial of the PKD polynomial in each spatial coordinate.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial.
        out (ndarray, optional): Shape (`N`, `d`) array to hold output.
        both: (bool, optional): If ``True``, return both the function value and
            gradient.

    Returns:
        If ``both=False``, an ndarray with shape (`N`, `d`) containing the
        gradient of the PKD polynomial with leading monomial degrees `i` at the
        points `x`.

        If ``both=True``, a tuple containing two ndarrays, one for the
        functional value and one for its gradient, in that order.
    '''
    if (d == 1):
        jnorm = jacobinorm2(i[0])**0.5
        if both:
            pkd = jacobi(i[0], x[:, 0]) / jnorm
        out = jacobider(i[0], x[:, 0]).reshape(x.shape)
        out /= jnorm
        if both:
            return (pkd, out)
        return out
    else:
        isum = sum(i[0:d-1])
        factor = (1. - x[:, d-1])
        tol = 1.e-10
        nonzero = np.abs(factor) > tol
        px = np.ndarray(x[:, 0:(d-1)].shape)
        px[nonzero, :] = ((x[nonzero, 0:(d-1)] + 1.) * 2. /
                          factor[nonzero, np.newaxis] - 1.)
        px[~nonzero, :] = -1.
        pxgradfisum = np.zeros(px.shape + (d,))
        for j in range(d-1):
            pxgradfisum[nonzero, j, j] = 2. * factor[nonzero]**(isum - 1)
            pxgradfisum[nonzero, j, d-1] = ((x[nonzero, j] + 1.) *
                                            2. * factor[nonzero]**(isum-2))
            if isum > 0:
                pxgradfisum[~nonzero, j, j] = 2. * factor[~nonzero]**(isum - 1)
                # otherwise, pigrad will be zero
            if isum > 1:
                pxgradfisum[~nonzero, j, d-1] = ((x[~nonzero, j] + 1.) * 2. *
                                                 factor[~nonzero]**(isum-2))
                # otherwise, pigrad * pzgrad will be independent of z
        pi, pigrad = proriolkoornwinderdubinergrad(d-1, i[0:d-1], px,
                                                   both=True)
        pz = jacobi(i[-1], x[:, d-1], 2*isum + d - 1, 0.)
        jnorm = jacobinorm2(i[-1], 2*isum+d-1, 0.)**0.5
        pz /= jnorm
        pzgrad = jacobider(i[-1], x[:, d-1], 2*isum + d - 1, 0.)
        pzgrad /= jnorm
        fisum = factor**isum
        if not (out is None):
            pkdgrad = out
            pkdgrad[:] = 0.
        else:
            pkdgrad = np.zeros(x.shape)
        if both:
            pkd = pi * pz * fisum * 2.**((d-1)/2)
        pkdgrad[:, :] = (np.einsum('ij,ijk->ik',
                                   pigrad,
                                   pxgradfisum).reshape(x.shape) *
                         pz[:, np.newaxis])
        pkdgrad[:, d-1] += pi * pzgrad * fisum
        if (isum != 0):
            pkdgrad[:, d-1] -= isum * pi * pz * factor**(isum-1)
        pkdgrad *= 2.**((d-1)/2)
        if both:
            return (pkd, pkdgrad)
        return pkdgrad


def proriolkoornwinderdubinerhess(d, i, x):
    '''Evaluation of the gradient of a Proriol-Koornwinder-Dubiner (PKD)
    polynomial.

    Args:
        d (int): The spatial dimension.
        i (tuple(int)): Multi-index of length `d`, the degree of the leading
            monomial of the PKD polynomial in each spatial coordinate.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial.

    Returns:
        ndarray: Shape (`N`, `d`, `d`) containing the Hessian of the PKD
        polynomial with leading monomial degrees `i` at the points `x`.
    '''
    if (d == 1):
        jnorm = jacobinorm2(i[0])**0.5
        out = jacobider(i[0], x[:, 0], k=2).reshape(x.shape + (1,))
        out /= jnorm
        return out
    else:
        isum = sum(i[0:d-1])
        factor = (1. - x[:, d-1])
        tol = 1.e-10
        nonzero = np.abs(factor) > tol
        px = np.ndarray(x[:, 0:(d-1)].shape)
        px[nonzero, :] = ((x[nonzero, 0:(d-1)] + 1.) * 2. /
                          factor[nonzero, np.newaxis] - 1.)
        px[~nonzero, :] = -1.
        pxgradf = np.zeros(px.shape + (d,))
        pxgradfisum = np.zeros(px.shape + (d,))
        pxgradfisumm1 = np.zeros(px.shape + (d,))
        pxhessfisum = np.zeros(px.shape + (d, d))
        for j in range(d-1):
            pxgradf[:, j, j] = 2.
            pxgradf[nonzero, j, d-1] = ((x[nonzero, j] + 1.) * 2. /
                                        factor[nonzero])
            pxgradfisum[nonzero, j, j] = 2. * factor[nonzero]**(isum - 1)
            pxgradfisum[nonzero, j, d-1] = ((x[nonzero, j] + 1.) * 2. *
                                            factor[nonzero]**(isum-2))
            if isum > 0:
                pxgradfisum[~nonzero, j, j] = 2. * factor[~nonzero]**(isum - 1)
                # otherwise, pigrad will be zero
            if isum > 1:
                pxgradfisum[~nonzero, j, d-1] = ((x[~nonzero, j] + 1.) * 2. *
                                                 factor[~nonzero]**(isum-2))
                # otherwise, pigrad * pzgrad will be independent of z
            pxgradfisumm1[nonzero, j, j] = 2. * factor[nonzero]**(isum - 2)
            pxgradfisumm1[nonzero, j, d-1] = ((x[nonzero, j] + 1.) * 2. *
                                              factor[nonzero]**(isum-3))
            if isum > 1:
                pxgradfisumm1[~nonzero, j, j] = 2. * factor[~nonzero]**(isum - 2)
                # otherwise, pigrad will be zero
            if isum > 2:
                pxgradfisumm1[~nonzero, j, d-1] = ((x[~nonzero, j] + 1.) * 2. *
                                                   factor[~nonzero]**(isum-3))
                # otherwise, pigrad * pzgrad will be independent of z
            pxhessfisum[nonzero, j, d-1, j] = 2. * factor[nonzero]**(isum-2)
            pxhessfisum[nonzero, j, j, d-1] = 2. * factor[nonzero]**(isum-2)
            pxhessfisum[nonzero, j, d-1, d-1] = ((x[nonzero, j] + 1.) * 4. *
                                                 factor[nonzero]**(isum-3))
            if isum > 1:
                pxhessfisum[~nonzero, j, d-1, j] = 2. * factor[~nonzero]**(isum-2)
                pxhessfisum[~nonzero, j, j, d-1] = 2. * factor[~nonzero]**(isum-2)
            if isum > 2:
                pxhessfisum[~nonzero, j, d-1, d-1] = ((x[~nonzero, j] + 1.)
                                                      * 4.
                                                      * factor[~nonzero]**(isum-3))
        pi, pigrad = proriolkoornwinderdubinergrad(d-1, i[0:d-1], px,
                                                   both=True)
        pihess = proriolkoornwinderdubinerhess(d-1, i[0:d-1], px)
        pz = jacobi(i[-1], x[:, d-1], 2*isum + d - 1, 0.)
        jnorm = jacobinorm2(i[-1], 2*isum+d-1, 0.)**0.5
        pz /= jnorm
        pzgrad = jacobider(i[-1], x[:, d-1], 2*isum + d - 1, 0.)
        pzgrad /= jnorm
        pzhess = jacobider(i[-1], x[:, d-1], 2*isum + d - 1, 0., k=2)
        pzhess /= jnorm
        fisum = factor**isum
        pkdhess = np.zeros(x.shape + (d,))
        pizgradfisum = np.einsum('ij,ijk->ik',
                                 pigrad,
                                 pxgradfisum).reshape(x.shape)
        pizgradfisumm1 = np.einsum('ij,ijk->ik',
                                   pigrad,
                                   pxgradfisumm1).reshape(x.shape)

        pkdhess += (np.einsum('ijl,ijk,ilm->ikm', pihess, pxgradf,
                              pxgradfisumm1).reshape(x.shape + (d,)) *
                    pz[:, np.newaxis, np.newaxis])
        pkdhess += (np.einsum('ij,ijkl->ikl',
                              pigrad,
                              pxhessfisum).reshape(x.shape + (d,)) *
                    pz[..., np.newaxis, np.newaxis])
        pkdhess[:, d-1, d-1] += pi * pzhess * fisum
        if (isum > 1):
            pkdhess[:, d-1, d-1] += (isum-1)*isum * pi * pz * factor**(isum-2)

        temp = pizgradfisum * pzgrad[..., np.newaxis]
        pkdhess[:, :, d-1] += temp
        pkdhess[:, d-1, :] += temp
        if (isum > 0):
            temp = pizgradfisumm1 * (pz * isum)[..., np.newaxis]
            pkdhess[:, :, d-1] -= temp
            pkdhess[:, d-1, :] -= temp
        if (isum > 0):
            pkdhess[:, d-1, d-1] -= 2. * pi * pzgrad * isum * factor**(isum-1)
        pkdhess *= 2.**((d-1)/2)
        return pkdhess


def proriolkoornwinderdubinervandermonde(d, n, x, out=None, C=None):
    '''Evaluation of the Vandermonde matrix of the Proriol-Koornwinder-Dubiner
    (PKD) polynomials.

    Args:
        d (int): The spatial dimension.
        n (int): The maximum degree of polynomials to include in the
            Vandermonde matrix.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial Vandermonde matrix.
        out (ndarray, optional): Shape (`N`, `\\binom{n+d}{d}`) array to hold
            output (or, if ``C`` is given, shape (`N`, `M`), where `M` is the
            number of columns of `C`).
        C (ndarray, optional): Shape (`\\binom{n+d}{d}`, `M`) matrix to
            multiply the PDK Vandermonde matrix by on the right.

    Returns:
        ndarray: Shape (`N`, `\\binom{n+d}{d}`), evaluation of the PKD
        polynomials with leading monomial degree at most `n` at the points `x`.
        The columns will index the PKD polynomials in lexicographic degree of
        their leading monomial exponents.

        Or, if `C` is given, shape (`N`, `M`), the product of the Vandermonde
        matrix with `C`.
    '''
    N = npolys(d, n)
    if out:
        V = out
    else:
        if C is None:
            V = np.ndarray(x.shape[0:-1]+(N,))
        else:
            V = np.ndarray(x.shape[0:-1]+(C.shape[1],))
    if not (C is None):
        V[:] = 0.
    for (k, i) in enumerate(multiindex_up_to(d, n)):
        if C is None:
            V[:, k] = proriolkoornwinderdubiner(d, i, x)
        else:
            V[:, :] += np.einsum('i,j->ij',
                                 proriolkoornwinderdubiner(d, i, x),
                                 C[k, :])
    return V


def proriolkoornwinderdubinervandermondegrad(d, n, x, out=None, C=None,
                                             work=None, both=False):
    '''Evaluation of the gradient of the Vandermonde matrix of the
    Proriol-Koornwinder-Dubiner (PKD) polynomials.

    Args:
        d (int): The spatial dimension.
        n (int): The maximum degree of polynomials to include in the
            Vandermonde matrix.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial Vandermonde matrix.
        out (ndarray, optional): Shape (`N`, `\\binom{n+d}{d}`, `d`) array to
            hold output (or, if ``C`` is given, shape (`N`, `M`, `d`), where
            `M` is the number of columns of `C`).
        C (ndarray, optional): Shape (`\\binom{n+d}{d}`, `M`) matrix to
            multiply the PDK Vandermonde matrix by on the right.

    Returns:
        ndarray: Shape (`N`, `\\binom{n+d}{d}`, `d`), evaluation of the
        gradient of the PKD Vandermonde matrix at the points `x`.

        Or, if `C` is given, shape (`N`, `M`, `d`), the gradient of the product
        of the Vandermonde matrix with `C`.
    '''
    N = npolys(d, n)
    if out:
        Vg = out
    else:
        if C is None:
            Vg = np.ndarray(x.shape[0:-1]+(N, d))
        else:
            Vg = np.ndarray(x.shape[0:-1]+(C.shape[1], d))
    if both:
        if C is None:
            V = np.ndarray(x.shape[0:-1]+(N,))
        else:
            V = np.ndarray(x.shape[0:-1]+(C.shape[1],))
    if both:
        V[:] = 0.
    Vg[:] = 0.
    if work is None:
        work = np.ndarray(x.shape[0:-1] + (d,))
    for (k, i) in enumerate(multiindex_up_to(d, n)):
        if both:
            v, work = proriolkoornwinderdubinergrad(d, i, x, out=work,
                                                    both=True)
        else:
            work = proriolkoornwinderdubinergrad(d, i, x, out=work)
        if C is None:
            if both:
                V[:, k] = v
                Vg[:, k, 0:d] = work
            else:
                Vg[:, k, 0:d] = work
        else:
            if both:
                V[:, :] += np.einsum('i,j->ij', v, C[k, :])
                Vg[:, :, :] += np.einsum('ij,k->ikj', work, C[k, :])
            else:
                Vg[:, :, :] += np.einsum('ij,k->ikj', work, C[k, :])
    if both:
        return (V, Vg)
    return Vg


def proriolkoornwinderdubinervandermondehess(d, n, x, C=None):
    '''Evaluation of the Hessian of the Vandermonde matrix of the
    Proriol-Koornwinder-Dubiner (PKD) polynomials.

    Args:
        d (int): The spatial dimension.
        n (int): The maximum degree of polynomials to include in the
            Vandermonde matrix.
        x (ndarray): Shape (`N`, `d`), points at which to evaluate the PDK
            polynomial Vandermonde matrix.
        C (ndarray, optional): Shape (`\\binom{n+d}{d}`, `M`) matrix to
            multiply the PDK Vandermonde matrix by on the right.

    Returns:
        ndarray: Shape (`N`, `\\binom{n+d}{d}`, `d`, `d`), evaluation of the
        Hessian of the PKD Vandermonde matrix at the points `x`.

        Or, if `C` is given, shape (`N`, `M`, `d`), the Hessian of the product
        of the Vandermonde matrix with `C`.
    '''
    N = npolys(d, n)
    if C is None:
        Vh = np.ndarray(x.shape[0:-1]+(N, d, d))
    else:
        Vh = np.ndarray(x.shape[0:-1]+(C.shape[1], d, d))
    Vh[:] = 0.
    for (k, i) in enumerate(multiindex_up_to(d, n)):
        work = proriolkoornwinderdubinerhess(d, i, x)
        if C is None:
            Vh[:, k, :, :] = work
        else:
            Vh[:, :, :, :] += np.einsum('ijk,l->iljk', work, C[k, :])
    return Vh
