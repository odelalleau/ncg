#!/usr/bin/env python

"""
Test code for non-linear conjugate gradient.
"""

__authors__ = "Olivier Delalleau, Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "BSD"
__contact__ = "Olivier Delalleau <delallea@iro>"


import math, sys, time
from itertools import islice

import miniml
import numpy

import theano
from theano import config, tensor
from ncg import leon_ncg


def get_data(spec):
    """
    Return iteratable on data specified by `spec`.
    """
    return eval('get_data_%s' % spec)()


def get_data_f1():
    """
    f1(x) = sin(pi * x) + normal(0, 0.01**2)

    x ~ U[-3, 3]
    """
    x_range = [-3, 3]
    noise = dict(mu=0, sigma=0.01)
    rng = get_rng()
    while True:
        x = rng.uniform(low=x_range[0], high=x_range[1])
        y = math.sin(math.pi * x) + rng.normal(loc=noise['mu'],
                                               scale=noise['sigma'])
        yield (numpy.array([x], dtype=config.floatX),
               numpy.array([y], dtype=config.floatX))


def get_model(spec):
    """
    Return model given by `spec`.

    :param spec: A string of the form "x-y-z-t" with `x` the size of the input
    layer, `t` the size of the output layer, and `y` and `z` the sizes of
    hidden layers. The number of hidden layers may be arbitrary.
    """
    sizes = spec.split('-')
    assert len(sizes) >= 2
    n_inputs = int(sizes[0])
    n_outputs = int(sizes[-1])
    n_hidden = map(int, sizes[1:-1])
    model = miniml.component.nnet.NNet(
            task='regression',
            n_units=n_hidden + [n_outputs],
            transfer_functions=['tanh'] * len(n_hidden) + ['identity'],
            hidden_transfer_function=None,
            n_hidden=None, n_out=None)
    return model


def get_rng(seed=None):
    if seed is None:
        seed = getattr(get_rng, 'seed', 1827)
        get_rng.seed = seed * 2 # for next RNG
    return numpy.random.RandomState(seed)


def minimize(model, data):
    pass


def test(data='f1', model='1-8-8-1', n_train=1000):
    data = list(islice(get_data(data), n_train))
    model = get_model(model)
    minimize(model, data)


def test_ncg_2(profile=True, pydot_print=True):

    rng = numpy.random.RandomState(232)

    all_vals = numpy.asarray(
        rng.uniform(size=(500*500,)),
        dtype=theano.config.floatX)

    idx  = 0
    vW0  = all_vals.reshape((500,500))

    vx  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)
    vy  = numpy.asarray(
        rng.uniform(size=(2000,500)), dtype=theano.config.floatX)

    W0 = theano.shared(vW0, 'W0')
    #W0  = tensor.specify_shape(_W0, vW0.shape)
    #W0.name = 'W0'

    x = theano.shared(vx, 'x')
    #x  = tensor.specify_shape(_x, vx.shape)
    #x.name = 'x'
    y = theano.shared(vy, 'y')
    #y  = tensor.specify_shape(_y, vy.shape)
    #y.name = 'y'

    def f(W0):
        return ((tensor.dot(x,W0) - y)**2).mean().mean()
        #return ((tensor.dot(x,W0) - y)**2).mean().mean() + abs(x).mean().mean()

    print 'Executing ncg'
    print '>>> Generating Graph'
    t0 = time.time()
    answers = leon_ncg(f, [W0], [], maxiter = 6,
                      profile = profile)
    tf = time.time() - t0

    print 'It took', tf, 'sec'
    print '>>> Compiling graph'
    t0 = time.time()
    func = theano.function([], answers, profile = profile,
                           name = 'test_fmincg_2',
                          mode = theano.Mode(linker='cvm'))
    tf = time.time() - t0
    print 'It took', tf, 'sec'
    if pydot_print:
        print '>>> Plotting graph'
    theano.printing.pydotprint(func,'t2_fmin_cg.png',
                           with_ids = True,
                           high_contrast = True,
                           scan_graphs = True)
    print 'Optimizing'
    t_th = 0
    t_py = 0
    for k in xrange(1):
        t0 = time.time()
        th_rval = func()[0]
        t_th += time.time() - t0

    print '-------- NOW SCIPY RESULTS ------'
    allw = tensor.vector('all')
    #allw  = tensor.specify_shape(_allw, all_vals.shape)
    idx  = 0
    W0   = allw.reshape((500,500))
    out    = f(W0)
    func   = theano.function([allw], out)
    gall   = tensor.grad(out, allw)
    fprime = theano.function([allw], gall)

    if pydot_print:
        theano.printing.pydotprint(func,  't2_f.png', with_ids = True,
                                   high_contrast = True)
        theano.printing.pydotprint(fprime,'t2_fprime.png', with_ids = True,
                                   high_contrast = True)

    # FIRST RUN with full_output to get an idea of how many steps where done
    t0 = time.time()
    rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                  maxiter = 6,
                                   full_output = 1,
                                    disp = 1)[1]

    t_py += time.time() - t0
    # rest runs with full_output 0
    '''
    for k in xrange(1):
        t0 = time.time()
        rval = py_fmin_cg(func, all_vals, fprime = fprime,
                                      maxiter = 6,
                                       full_output = 1,
                                       disp = 0 )[1]

        t_py += time.time() - t0
    '''
    print 'THEANO output :: ',th_rval
    print 'NUMPY  output :: ',rval
    print
    print 'Timings'
    print
    print 'theano ---------> time %e'% t_th
    print 'numpy  ---------> time %e'% t_py


def main():
    test()
    #test_ncg_2()
    return 0


if __name__ == '__main__':
    sys.exit(main())
