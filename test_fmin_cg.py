#!/usr/bin/env python

"""
Test code for non-linear conjugate gradient.
"""

__authors__ = "Olivier Delalleau, Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "BSD"
__contact__ = "Olivier Delalleau <delallea@iro>"


import math, sys, time
from itertools import islice, izip

import ubi_mm
ubi_mm.init_ml(__file__)
import ml
import ml.util

import miniml
import numpy
import scipy
from matplotlib import pyplot

import theano
from theano import config, tensor
from ncg import leon_ncg_python


class ModelInterface(object):

    """
    Provides an interface with convenience functions for optimization.
    """

    def __init__(self, model, data):
        self.model = model
        params = self.model.params
        n_params = sum(p.get_value(borrow=True).size for p in params)
        self.params_vec = numpy.zeros(n_params, dtype=config.floatX)
        self.compute_cost = theano.function(
                [self.model.input, self.model.task_spec.target],
                self.model.task_spec.total_cost)
        self.compute_grad = theano.function(
                [self.model.input, self.model.task_spec.target],
                tensor.grad(self.model.task_spec.total_cost, params))
        self.compute_output = self.model.task_spec.compute_output

        # Build data matrices.
        n_samples = len(data)
        first_input = data[0][0]
        if first_input.shape:
            assert len(first_input.shape) == 1
            input_size = len(first_input)
        else:
            # Scalar value.
            input_size = 1
        first_target = data[0][1]
        if first_target.shape:
            assert len(first_target.shape) == 1
            target_size = len(first_target)
        else:
            # Scalar value.
            target_size = 1
        print 'input_size = %s, target_size = %s' % (input_size, target_size)
        self.data_input = numpy.zeros((n_samples, input_size),
                                      dtype=config.floatX)
        self.data_target = numpy.zeros((n_samples, target_size),
                                       dtype=config.floatX)
        for i, sample in enumerate(data):
            input, target = sample
            self.data_input[i] = input
            self.data_target[i] = target

    def fill_params(self, param_values):
        """
        Fill parameters with provided values.
        """
        idx = 0
        for p in self.model.params:
            p_current = p.get_value(borrow=True)
            p_vals = param_values[idx:idx + p_current.size]
            p.set_value(p_vals.reshape(p_current.shape))
            idx += p_current.size

    def all_costs(self, param_values):
        """
        Return all costs at given parameter values.
        """
        self.fill_params(param_values)
        return self.model.task_spec.compute_costs(self.data_input, self.data_target)

    def cost(self, param_values):
        """
        Return main cost at given parameter values.
        """
        self.fill_params(param_values)
        return self.compute_cost(self.data_input, self.data_target)

    def grad(self, param_values):
        """
        Return gradient at given parameter values.
        """
        self.fill_params(param_values)
        grads = self.compute_grad(self.data_input, self.data_target)
        return self.flatten(grads)

    def flatten(self, arrays):
        """
        Return a vector containing all elements in all arrays.

        The total number of elements is assumed to be the number of float
        parameters to be optimized.
        """
        rval = self.params_vec.copy()
        # Fill vector with content of all arrays.
        idx = 0
        for array_val in arrays:
            rval[idx:idx + array_val.size] = array_val.flatten()
            idx += array_val.size
        return rval

    def get_minibatch(self, k0, k1):
        """
        Return the pair (input, target) for minibatch [k0:k1].
        """
        if k1 is None:
            # Easy case, get all data until end.
            return self.data_input[k0:], self.data_target[k0:]
        minibatch_size = k1 - k0
        assert minibatch_size > 0
        n_samples = len(self.data_input)
        # Get index of first sample.
        k0 = k0 % n_samples
        # Get upper range value.
        k1 = k0 + minibatch_size

        # Handle simple situation with no cycle.
        if k1 <= n_samples:
            return self.data_input[k0:k1], self.data_target[k0:k1]

        # Build minibatch.
        rval_input = numpy.zeros((minibatch_size, self.data_input.shape[1]),
                                 dtype=self.data_input.dtype)
        rval_target = numpy.zeros((minibatch_size, self.data_target.shape[1]),
                                  dtype=self.data_target.dtype)
        start_minibatch = 0
        start_data = k0
        n_left = minibatch_size
        while n_left > 0:
            #print 'start_minibatch = %s' % start_minibatch
            #print 'start_data = %s' % start_data
            n_to_add = min(n_left, n_samples - start_data)
            rval_input[start_minibatch:start_minibatch + n_to_add] = (
                            self.data_input[start_data:start_data + n_to_add])
            rval_target[start_minibatch:start_minibatch + n_to_add] = (
                            self.data_target[start_data:start_data + n_to_add])
            n_left -= n_to_add
            start_minibatch += n_to_add
            start_data = (start_data + n_to_add) % n_samples
        return rval_input, rval_target

    def make_cost(self, k0, k1):
        """
        Return callable function to compute cost on minibatch [k0:k1].
        """
        input, target = self.get_minibatch(k0, k1)
        def f(param_values):
            self.fill_params(param_values)
            return self.compute_cost(input, target)
        return f

    def make_grad(self, k0, k1):
        """
        Return callable function to compute gradient on minibatch [k0:k1].
        """
        input, target = self.get_minibatch(k0, k1)
        def g(param_values):
            self.fill_params(param_values)
            grads = self.compute_grad(input, target)
            return self.flatten(grads)
        return g

    def params_to_vec(self):
        return self.flatten([p.get_value(borrow=True)
                             for p in self.model.params])


def as_array(*args):
    return [numpy.asarray(x, dtype=config.floatX) for x in args]


def get_data(spec):
    """
    Return iteratable on data specified by `spec`.
    """
    if '(' in spec:
        start = spec.find('(')
        end = spec.find(')')
        assert start > 0 and end > 0
        args = spec[start + 1:end]
    else:
        args = ''
        start = len(spec)
    return eval('get_data_%s(%s)' % (spec[0:start], args))


def get_data_f1():
    """
    f1(x) = sin(pi * x) + normal(0, 0.1**2)

    x ~ U[-3, 3]
    """
    x_range = [-3, 3]
    noise = dict(mu=0, sigma=0.1)
    rng = get_rng()
    while True:
        x = rng.uniform(low=x_range[0], high=x_range[1])
        y = math.sin(math.pi * x) + rng.normal(loc=noise['mu'],
                                               scale=noise['sigma'])
        yield as_array(x, y)


def get_data_f2():
    """
    f2(x) = (x - 1)**2

    x ~ U[-10, 10]
    """
    x_range = [-10, 10]
    rng = get_rng()
    while True:
        x = rng.uniform(low=x_range[0], high=x_range[1])
        y = (x - 1)**2
        yield as_array(x, y)


def get_data_f3(d):
    """
    f3(x) = w' x + b + epsilon

    x ~ U[-3, 3]^d
    w = [1/d, 2/d, ..., 1]
    b = 1
    epsilon ~ N(0, 0.1)
    """
    x_range = [-3, 3]
    rng = get_rng()
    w = numpy.arange(1, d + 1) / float(d)
    b = 1.
    while True:
        x = rng.uniform(low=x_range[0], high=x_range[1], size=d)
        epsilon = rng.normal(loc=0, scale=0.1)
        y = numpy.dot(w, x) + b + epsilon
        yield as_array(x, y)


def get_model(spec, data):
    """
    Return model given by `spec`.

    :param spec: A string of the form "x-y-z-t" with `x` the size of the input
    layer, `t` the size of the output layer, and `y` and `z` the sizes of
    hidden layers. The number of hidden layers may be arbitrary.
    By default the transfer function of hidden layers is `tanh`, while the
    transfer function of the output layer is `identity`. These may be changed
    by specifying the transfer function within parenthesis, for instance:
        3-5-6(sigmoid)-3(identity)-1(sigmoid)
    """
    def parse_size(s, default_transfer_function):
        """
        Return the pair (size, transfer_function) corresponding to string `s`.
        """
        if '(' in s:
            start = s.find('(')
            end = s.find(')')
            assert start > 0 and end > 0
            transfer_function = s[start + 1:end]
        else:
            transfer_function = default_transfer_function
            start = len(s)
        return int(s[0:start]), transfer_function

    sizes = spec.split('-')
    assert len(sizes) >= 2
    n_inputs, _ = parse_size(sizes[0], None)
    assert _ is None    # No transfer function on inputs.
    n_outputs, output_transfer = parse_size(sizes[-1], 'identity')
    hidden = map(parse_size, sizes[1:-1], ['tanh'] * (len(sizes) - 2))
    n_hidden = [h[0] for h in hidden]
    hidden_transfer = [h[1] for h in hidden]
    nnet = miniml.component.nnet.NNet(
            task='regression',
            n_units=n_hidden + [n_outputs],
            transfer_functions=hidden_transfer + [output_transfer],
            hidden_transfer_function=None,
            n_hidden=None, n_out=None)
    
    # Properly initialize all weights.
    w0v = nnet.weights[0].get_value(borrow=True)
    n_hidden_1 = w0v.shape[1]
    nnet.weights[0].set_value(numpy.zeros((n_inputs, n_hidden_1),
                                          dtype=config.floatX))
    nnet.seed = nnet.get_seed(n_inputs)
    nnet.forget()
    nnet.init_weights()

    # Currently we ignore those, so better make sure they are not used.
    assert not nnet.task_spec.new_params
    assert nnet.regularization_coeff == 0
    assert nnet.output_is_layer == -1

    # Gather list of all parameters.
    params = nnet.weights + nnet.biases

    # Expose model interface.
    model = miniml.utility.Storage(
            task_spec=nnet.task_spec,
            input=nnet.input,
            params=params)
    ui = ModelInterface(model=model, data=data)
    return ui


def get_rng(seed=None):
    if seed is None:
        seed = getattr(get_rng, 'seed', 1827)
        get_rng.seed = seed * 2 # for next RNG
    return numpy.random.RandomState(seed)


def minimize(model, data):
    best = [None]
    count = [0]
    errors = []
    lambdas = []
    def callback(param_values, lambda_t):
        count[0] += 1
        cost = model.all_costs(param_values)
        print '%s: %s' % (count[0], cost['mse'])
        best[0] = param_values
        errors.append(cost['mse'])
        lambdas.append(lambda_t)
    leon_ncg_python(
            make_f=model.make_cost,
            w_0=model.params_to_vec(),
            make_fprime=model.make_grad,
            callback=callback,
            maxiter=1000,
            #direction='polak-ribiere',
            direction='hestenes-stiefel',
            minibatch_size=100,
            minibatch_offset=30,
            )
    return best[0], errors, lambdas


def plot(model, params, errors, lambdas):
    """
    Plot:
        - true data vs. prediction
        - training error over time
        - evolution of lambda_t
    """
    to_plot = []
    model.fill_params(params)
    model_output = model.compute_output(model.data_input,
                                        model.data_target)

    for input, target, output in izip(model.data_input,
                                      model.data_target,
                                      model_output):
        to_plot.append([input[0], target[0], output[0]])

    to_plot = numpy.array(sorted(to_plot))

    # Output.
    fig = pyplot.figure()
    pyplot.plot(to_plot[:, 0], to_plot[:, 1], label='true')
    pyplot.plot(to_plot[:, 0], to_plot[:, 2], label='model')
    pyplot.legend()

    # Training error.
    fig = pyplot.figure()
    pyplot.plot(errors)
    pyplot.xlabel('time')
    pyplot.ylabel('training error')

    # Evolution of lambda_t.
    fig = pyplot.figure()
    pyplot.plot(lambdas)
    pyplot.xlabel('k')
    pyplot.ylabel('lambda_t')

    # Show plots.
    pyplot.show()


# TODO n_train=1000 WORKS MUCH BETTER THAN n_train=999 => SUSPICIOUS
def test(data_spec='f3(1000)', model_spec='1000-1', n_train=1000):
    data = list(islice(get_data(data_spec), n_train))
    model = get_model(spec=model_spec, data=data)
    params, errors, lambdas = minimize(model, data)
    plot(model, params, errors, lambdas)


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
    ml.util.run_with_try(_main)

def _main():
    test()
    #test_ncg_2()
    return 0


if __name__ == '__main__':
    sys.exit(main())
