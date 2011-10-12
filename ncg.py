"""
Experimental non-linear conjugate gradient.
"""

__authors__ = "Olivier Delalleau, Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "BSD"
__contact__ = "Olivier Delalleau <delallea@iro>"


from itertools import izip

import numpy
from scipy.optimize.optimize import (
        _epsilon, line_search_wolfe1, line_search_wolfe2, vecnorm,
        wrap_function)

import theano
import theano.tensor as TT
from theano.lazycond import ifelse
from theano.scan_module import until

from pylearn2.optimization.ncg import linesearch_module as linesearch
from pylearn2.optimization.ncg.ncg_module import (
        lazy_or, zero)


def leon_ncg_theano(cost_fn, x0s, args=(), gtol=1e-5,
        maxiter=None, profile=False):
    """
    Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    cost_fn : callable f(*(xs+args))
        Objective function to be minimized.
    x0s : list of theano tensors
        Initial guess.
    args : tuple
        Extra arguments passed to cost_fn.
    gtol : float
        Stop when norm of gradient is less than gtol.
    maxiter : int
        Maximum number of iterations allowed for CG
    profile: flag (boolean)
        If profiling information should be printed

    Returns
    -------
    fopt : float
        Minimum value found, f(xopt).
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the nonlinear conjugate gradient algorithm of Polak and
    Ribiere. See Wright & Nocedal, 'Numerical Optimization',
    1999, pg. 120-122.

    This function mimics `fmin_cg` from `scipy.optimize`.
    """
    if type(x0s) not in (tuple, list):
        x0s = [x0s]
    else:
        x0s = list(x0s)
    if type(args) not in (tuple, list):
        args = [args]
    else:
        args = list(args)


    if maxiter is None:
        len_x0 = sum(x0.size for x0 in x0s)
        maxiter = len_x0 * 200

    out = cost_fn(*(x0s+args))
    global_x0s = [x for x in x0s]
    def f(*nw_x0s):
        rval = theano.clone(out, replace=dict(zip(global_x0s, nw_x0s)))
        #rval = cost_fn(*nw_x0s)
        return rval


    def myfprime(*nw_x0s):
        gx0s = TT.grad(out, global_x0s, keep_wrt_type=True)
        rval = theano.clone(gx0s, replace=dict(zip(global_x0s, nw_x0s)))
        #import ipdb; ipdb.set_trace()
        #rval = TT.grad(cost_fn(*nw_x0s), nw_x0s)
        return [x for x in rval]


    n_elems = len(x0s)

    def fmin_cg_loop(old_fval, old_old_fval, *rest):
        xks  = rest[:n_elems]
        gfks = rest[n_elems:n_elems * 2]

        maxs = [ abs(gfk).max(axis=range(gfk.ndim)) for gfk in gfks ]
        if len(maxs) == 1:
            gnorm = maxs[0]
        else:
            gnorm = TT.maximum(maxs[0], maxs[1])
            for dx in maxs[2:]:
                gnorm = TT.maximum(gnorm, dx)

        pks  = rest[n_elems*2:]
        #import ipdb; ipdb.set_trace()
        deltak = sum((gfk * gfk).sum() for gfk in gfks)

        old_fval_backup = old_fval
        old_old_fval_backup = old_old_fval

        alpha_k, old_fval, old_old_fval, derphi0, nw_gfks = \
                linesearch.line_search_wolfe2(f,myfprime, xks, pks,
                                              old_fval_backup,
                                              old_old_fval_backup,
                                              profile = profile,
                                             gfks = gfks)



        xks = [ ifelse(gnorm <= gtol, xk,
                              ifelse(TT.bitwise_or(TT.isnan(alpha_k),
                                                          TT.eq(alpha_k,
                                                                zero)), xk,
                                            xk+alpha_k*pk)) for xk, pk in zip(xks,pks)]
        gfkp1s_tmp = myfprime(*xks)
        gfkp1s = [ ifelse(TT.isnan(derphi0), nw_x, x) for nw_x, x in
                  zip(gfkp1s_tmp, nw_gfks)]


        yks = [gfkp1 - gfk for gfkp1, gfk in izip(gfkp1s, gfks)]
        # Polak-Ribiere formula.
        beta_k = TT.maximum(
                zero,
                sum((x * y).sum() for x, y in izip(yks, gfkp1s)) / deltak)
        pks  = [ ifelse(gnorm <= gtol, pk,
                               ifelse(TT.bitwise_or(TT.isnan(alpha_k),
                                                           TT.eq(alpha_k,
                                                                 zero)), pk, -gfkp1 +
                                             beta_k * pk)) for gfkp1,pk in zip(gfkp1s,pks) ]
        gfks = [ifelse(gnorm <= gtol,
                       gfk,
                       ifelse(
                           TT.bitwise_or(TT.isnan(alpha_k),
                                         TT.eq(alpha_k, zero)),
                           gfk,
                           gfkp1))
                for (gfk, gfkp1) in izip(gfks, gfkp1s)]

        stop = lazy_or(gnorm <= gtol, TT.bitwise_or(TT.isnan(alpha_k),
                                                TT.eq(alpha_k, zero)))# warnflag = 2
        old_fval     = ifelse(gnorm >gtol, old_fval, old_fval_backup)
        old_old_fval = ifelse(gnorm >gtol, old_old_fval,
                                     old_old_fval_backup)
        return ([old_fval, old_old_fval]+xks + gfks + pks,
                until(stop))

    gfks = myfprime(*x0s)
    xks = x0s
    old_fval = f(*xks)

    old_old_fval = old_fval + numpy.asarray(5000, dtype=theano.config.floatX)

    old_fval.name = 'old_fval'
    old_old_fval.name = 'old_old_fval'
    pks = [-gfk for gfk in gfks]

    outs, _ = theano.scan(fmin_cg_loop,
                          outputs_info = [old_fval,
                                          old_old_fval] + xks + gfks + pks,
                          n_steps = maxiter,
                          name = 'fmin_cg',
                          mode = theano.Mode(linker='cvm_nogc'),
                          profile = profile)

    sol = [outs[0][-1]] + [x[-1] for x in outs[2:2+n_elems]]
    return sol

def leon_ncg_python(make_f, w_0, make_fprime=None, gtol=1e-5, norm=numpy.Inf,
              maxiter=None, full_output=0, disp=1, retall=0, callback=None,
              direction='hestenes-stiefel',
              minibatch_size=None,
              minibatch_offset=None,
              ):
    """Minimize a function using a nonlinear conjugate gradient algorithm.

    Parameters
    ----------
    make_f : callable make_f(k0, k1)
    When called with (k0, k1) as arguments, return a function f such that
    f(w) is the objective to be minimize at parameter w, on minibatch x_k0
    to x_k1. If k1 is None then the minibatch should contain all the
    remaining data.
    w_0 : ndarray
    Initial guess.
    make_fprime : callable make_f'(k0, k1)
    Same as `make_f`, but to compute the derivative of f on a minibatch.
    gtol : float
    Stop when norm of gradient is less than gtol.
    norm : float
    Order of vector norm to use. -Inf is min, Inf is max.
    size (can be scalar or vector).
    callback : callable
    An optional user-supplied function, called after each
    iteration. Called as callback(w_t, lambda_t), where w_t is the
    current parameter vector and lambda_t the coefficient for the
    new direction.
    direction : string
    Formula used to computed the new direction, among:
        - polak-ribiere
        - hestenes-stiefel
    minibatch_size : int
    Size of each minibatch. Use None for batch learning.
    minibatch_offset : int
    Shift of the minibatch. Use None to use the minibatch size (i.e. no
    overlap at all if the minibatch size is less than half of the whole
    dataset size).

    Returns
    -------
    xopt : ndarray
    Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
    Minimum value found, f(xopt).
    func_calls : int
    The number of function_calls made.
    grad_calls : int
    The number of gradient calls made.
    warnflag : int
    1 : Maximum number of iterations exceeded.
    2 : Gradient and/or function calls not changing.
    allvecs : ndarray
    If retall is True (see other parameters below), then this
    vector containing the result at each iteration is returned.

    Other Parameters
    ----------------
    maxiter : int
    Maximum number of iterations to perform.
    full_output : bool
    If True then return fopt, func_calls, grad_calls, and
    warnflag in addition to xopt.
    disp : bool
    Print convergence message if True.
    retall : bool
    Return a list of results at each iteration if True.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the nonlinear conjugate gradient algorithm of Polak and
    Ribiere. See Wright & Nocedal, 'Numerical Optimization',
    1999, pg. 120-122.
    """
    if minibatch_offset is None:
        if minibatch_size is None:
            # Batch learning: no offset is needed.
            minibatch_offset = 0
        else:
            # Use the same offset as the minibatch size.
            minibatch_offset = minibatch_size
    w_0 = numpy.asarray(w_0).flatten()
    if maxiter is None:
        maxiter = len(w_0)*200
    k0 = 0
    k1 = minibatch_size
    f = make_f(k0, k1)
    assert make_fprime is not None
    fprime = make_fprime(k0, k1)
    func_calls = [0]
    grad_calls = [0]
    tmp_func_calls, f = wrap_function(f, ())
    tmp_grad_calls, myfprime = wrap_function(fprime, ())
    g_t = myfprime(w_0)
    t = 0
    N = len(w_0)
    w_t = w_0
    old_fval = f(w_t)
    old_old_fval = old_fval + 5000

    if retall:
        allvecs = [w_t]
    warnflag = 0
    d_t = -g_t
    gnorm = vecnorm(g_t, ord=norm)

    while (gnorm > gtol) and (t < maxiter):
        # These values are modified by the line search, even if it fails
        old_fval_backup = old_fval
        old_old_fval_backup = old_old_fval

        alpha_t, fc, gc, old_fval, old_old_fval, h_t = \
                 line_search_wolfe1(f, myfprime, w_t, d_t, g_t, old_fval,
                                  old_old_fval, c2=0.4)
        if alpha_t is None: # line search failed -- use different one.
            alpha_t, fc, gc, old_fval, old_old_fval, h_t = \
                     line_search_wolfe2(f, myfprime, w_t, d_t, g_t,
                                        old_fval_backup, old_old_fval_backup)
            if alpha_t is None or alpha_t == 0:
                # This line search also failed to find a better solution.
                raise AssertionError()
                warnflag = 2
                break
        # Update weights.
        w_tp1 = w_t + alpha_t * d_t

        # Compute derivative after the weight update, if not done already.
        if h_t is None:
            h_t = myfprime(w_tp1)
        else:
            assert (h_t == myfprime(w_tp1)).all() # Sanity check.

        # Switch to next minibatch.
        func_calls[0] += tmp_func_calls[0]
        grad_calls[0] += tmp_grad_calls[0]
        k0 += minibatch_offset
        if minibatch_size is None:
            k1 = None
        else:
            k1 = k0 + minibatch_size
        tmp_func_calls, f = wrap_function(make_f(k0, k1), ())
        tmp_grad_calls, myfprime = wrap_function(make_fprime(k0, k1), ())

        # Compute derivative on new minibatch.
        g_tp1 = myfprime(w_tp1)

        if retall:
            allvecs.append(w_tp1)
        h_t_minus_g_t = h_t - g_t
        if direction == 'polak-ribiere':
            # Polak-Ribiere.
            delta_t = numpy.dot(g_t, g_t)
            lambda_t = max(0, numpy.dot(h_t_minus_g_t, g_tp1) / delta_t)
        elif direction == 'hestenes-stiefel':
            # Hestenes-Stiefel.
            lambda_t = max(0, numpy.dot(h_t_minus_g_t, g_tp1) / numpy.dot(h_t_minus_g_t, d_t))
        else:
            raise NotImplementedError(direction)
        d_t = -g_tp1 + lambda_t * d_t
        g_t = g_tp1
        w_t = w_tp1
        gnorm = vecnorm(g_t, ord=norm)
        if callback is not None:
            callback(w_t, lambda_t)
        t += 1


    if disp or full_output:
        fval = old_fval
    if warnflag == 2:
        if disp:
            print "Warning: Desired error not necessarily achieved due to precision loss"
            print " Current function value: %f" % fval
            print " Iterations: %d" % t
            print " Function evaluations: %d" % func_calls[0]
            print " Gradient evaluations: %d" % grad_calls[0]

    elif t >= maxiter:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
            print " Current function value: %f" % fval
            print " Iterations: %d" % t
            print " Function evaluations: %d" % func_calls[0]
            print " Gradient evaluations: %d" % grad_calls[0]
    else:
        if disp:
            print "Optimization terminated successfully."
            print " Current function value: %f" % fval
            print " Iterations: %d" % t
            print " Function evaluations: %d" % func_calls[0]
            print " Gradient evaluations: %d" % grad_calls[0]


    if full_output:
        retlist = w_t, fval, func_calls[0], grad_calls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = w_t
        if retall:
            retlist = (w_t, allvecs)

    return retlist



