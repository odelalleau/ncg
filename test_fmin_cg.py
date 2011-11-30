#!/usr/bin/env python

"""
Test code for non-linear conjugate gradient.
"""

__authors__ = "Olivier Delalleau, Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "BSD"
__contact__ = "Olivier Delalleau <delallea@iro>"


import math, os, sys, time
from itertools import islice, izip

import matplotlib
import ubi_mm
ubi_mm.init_ml(__file__)
import ml
import ml.util

import dlt # Deep Learning Tutorials
from dlt.logistic_sgd import load_data
miniml = None # Lazy import
import numpy
import scipy

import theano
from theano import config, tensor
from ncg import leon_ncg_python


class ModelInterface(object):

    """
    Provides an interface with convenience functions for optimization.
    """

    def __init__(self, model, data_iter, n_offline_train, n_test, task):
        self.model = model
        self.data_iter = data_iter
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
        data = {}
        data['test'] = list(islice(self.data_iter, n_test))
        data['offline_train'] = list(islice(self.data_iter, n_offline_train))
        first_input = data['offline_train'][0][0]
        if first_input.shape:
            assert len(first_input.shape) == 1
            input_size = len(first_input)
        else:
            # Scalar value.
            input_size = 1
        first_target = data['offline_train'][0][1]
        if first_target.shape:
            assert len(first_target.shape) == 1
            target_size = len(first_target)
        else:
            # Scalar value.
            target_size = 1
        print 'input_size = %s, target_size = %s' % (input_size, target_size)

        n_data = {'offline_train': n_offline_train,
                  'test': n_test,
                  'online_train': n_offline_train,
                  }
        self.data_input = {}
        self.data_target = {}
        for data_type in n_data:
            self.data_input[data_type] = numpy.zeros(
                    (n_data[data_type], input_size), dtype=config.floatX)
            if task == 'classification':
                target_dtype = 'int64'
                target_shape = (n_data[data_type],)
                assert target_size == 1
            else:
                assert task == 'regression'
                target_dtype = config.floatX
                target_shape = (n_data[data_type], target_size)
            self.data_target[data_type] = numpy.zeros(
                    target_shape, dtype=target_dtype)
            for i, sample in enumerate(data.get(data_type, [])):
                input, target = sample
                self.data_input[data_type][i] = input
                self.data_target[data_type][i] = target
        # Copy offline train data as first chunk of online data.
        for d in self.data_input, self.data_target:
            d['online_train'][:] = d['offline_train']
        # Store range of the current chunk of online data.
        self.online_chunk = [0, n_data['online_train']]

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
        return [self.model.task_spec.compute_costs(self.data_input[t],
                                                   self.data_target[t])
                for t in ('offline_train', 'test')]

    def cost(self, param_values):
        """
        Return main cost at given parameter values.
        """
        raise AssertionError('We are not currently using this function')
        self.fill_params(param_values)
        return self.compute_cost(self.data_input, self.data_target)

    def grad(self, param_values):
        """
        Return gradient at given parameter values.
        """
        raise AssertionError('We are not currently using this function')
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

        If k1 is None then we use offline training data.
        """
        print 'get_minibatch(%s, %s)' % (k0, k1)
        input = self.data_input['online_train']
        target = self.data_target['online_train']
        if k1 is None:
            # Easy case, get offline training data.
            # Currently this should only happen when k0 == 0.
            assert k0 == 0
            #print '%s -> %s' % (input[k0][0], input[-1][0])
            return input[k0:], target[k0:]

        # Ensure our data chunk can store the full minibatch being requested.
        assert k1 - k0 <= len(input)

        # Ensure we are not trying to go back in time.
        assert k0 >= self.online_chunk[0]

        if k1 > self.online_chunk[1]:
            # Need to retrieve more data from iterator.
            # First copy the data already available.
            start = k0 - self.online_chunk[0]
            size = len(input) - start
            for d in input, target:
                d[0:size] = d[start:start + size].copy()
            # Then retrieve more data.
            #print 'Retrieving %s more samples' % start
            for i, sample in enumerate(islice(self.data_iter, start)):
                input[size + i] = sample[0]
                target[size + i] = sample[1]
            # And update online chunk info.
            self.online_chunk[0] = k0
            self.online_chunk[1] = k0 + len(input)

        start = k0 - self.online_chunk[0]
        end = k1 - self.online_chunk[0]
        #print '%s -> %s' % (input[start][0], input[end - 1][0])
        return input[start:end], target[start:end]

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
    rng = get_rng(seed=8948394)
    w = numpy.arange(1, d + 1) / float(d)
    b = 1.
    while True:
        x = rng.uniform(low=x_range[0], high=x_range[1], size=d)
        epsilon = rng.normal(loc=0, scale=0.1)
        y = numpy.dot(w, x) + b + epsilon
        yield as_array(x, y)


def get_data_f4():
    """
    f4(x) = x

    x increases by 1 at each sample, starting at 0.
    """
    x = 0.
    while True:
        yield as_array([x], [x / 1000])
        x += 1


def get_data_mnist(n_train, n_valid, n_test):
    """
    MNIST dataset.

    Return first the validation samples, then the test samples, then keep
    iterating on the training samples.
    """
    dlt_dir = dlt.__path__[0]
    mnist_dataset = os.path.realpath(os.path.join(dlt_dir, '..', 'data',
                                                  'mnist.pkl.gz'))
    assert os.path.exists(mnist_dataset)
    datasets = load_data(mnist_dataset)
    get_data = []
    idx = tensor.lscalar('idx')
    for data in datasets:
        get_data.append(theano.function([idx], [data[0][idx], data[1][idx]]))
    # First yield validation and test samples.
    for dataset_idx, n_samples in ((1, n_valid), (2, n_test)):
        for i in xrange(n_samples):
            sample = get_data[dataset_idx](i)
            yield as_array(*sample)
    # Then iterate on training samples.
    i = 0
    while True:
        sample = get_data[0](i)
        yield as_array(*sample)
        i = (i + 1) % n_train


def get_model(spec, **args):
    """
    Return model given by `spec`.

    :param spec: A string of the form "x-y-z-t" with `x` the size of the input
    layer, `t` the size of the output layer, and `y` and `z` the sizes of
    hidden layers. The number of hidden layers may be arbitrary.
    By default the transfer function of hidden layers is `tanh`, while the
    transfer function of the output layer is `identity`. These may be changed
    by specifying the transfer function within parenthesis, for instance:
        3-5-6(sigmoid)-3(identity)-1(sigmoid)

    :param args: Arguments forwarded to ModelInterface.
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
    task = args['task']
    nnet = miniml.component.nnet.NNet(
            task=task,
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
    params = nnet.params

    # Expose model interface.
    model = miniml.utility.Storage(
            task_spec=nnet.task_spec,
            input=nnet.input,
            params=params)
    ui = ModelInterface(model=model, **args)
    return ui


def get_rng(seed=None):
    if seed is None:
        seed = getattr(get_rng, 'seed', 1827)
        get_rng.seed = seed * 2 # for next RNG
    return numpy.random.RandomState(seed)


def minimize(model, task, **args):
    best = [None]
    count = [0]
    lambdas = []
    if task == 'regression':
        cost_names = ['mse']
    elif task == 'classification':
        cost_names = ['nll', 'class_error']
    else:
        raise NotImplementedError(task)
    errors = dict((c, []) for c in cost_names)
    def callback(param_values, lambda_t):
        count[0] += 1
        cost = model.all_costs(param_values)
        print '%s: %s (%s)' % (count[0],
                               ', '.join('%.4f' % c[cost_names[0]] for c in cost),
                               param_values[0:3])
        best[0] = param_values
        for cname in cost_names:
            errors[cname].append([c[cname] for c in cost])
        lambdas.append(lambda_t)
    leon_ncg_python(
            make_f=model.make_cost,
            w_0=model.params_to_vec(),
            make_fprime=model.make_grad,
            callback=callback,
            #direction='polak-ribiere',
            direction='hestenes-stiefel',
            **args
            )
    return best[0], errors, lambdas, args['minibatch_size']


def plot(results, experiments, show_plots=True, expdir=None):
    """
    Plot:
        - true data vs. prediction
        - training and test error over time
        - evolution of lambda_t
    """
    # Should we display on screen or just save as a file?
    import matplotlib.pyplot as pyplot
    if show_plots:
        plot_ext = 'png'
    else:
        plot_ext = 'pdf'

    # Model output (currently disabled).
    if False:
        to_plot = []
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

    # Find maximum number of samples.
    max_n_samples = 0

    for exp_name, model, params, errors, lambdas, minibatch_size in results:
        if minibatch_size is None:
            # Offline batch setting.
            minibatch_size = len(model.data_input['offline_train'])
        max_n_samples = max(max_n_samples, len(errors[errors.keys()[0]]) * minibatch_size)

    def complete(lst, n):
        if len(lst) < n:
            lst += [lst[-1]] * (n - len(lst))

    def get_figure(i):
        if show_plots:
            return pyplot.figure(i)
        else:
            return pyplot.figure(i, figsize=(15, 15))

    for exp_name, model, params, errors, lambdas, minibatch_size in results:

        model.fill_params(params)
        # Figure out x axis (number of samples visited).
        if minibatch_size is None:
            # Offline batch setting.
            minibatch_size = len(model.data_input['offline_train'])
        x_vals = range(minibatch_size, max_n_samples + 1,
                       minibatch_size)

        fig_idx = 1
        for cname in sorted(errors):
            # Offline training error.
            fig = get_figure(fig_idx)
            fig_idx += 1
            to_plot = [e[0] for e in errors[cname]]
            complete(to_plot, len(x_vals))
            pyplot.plot(x_vals, to_plot, label=exp_name)
            if False:
                # Debug indicators of restarts.
                for xv, lamb in izip(x_vals, lambdas):
                    if lamb == 0:
                        pyplot.axvline(x=xv)

            # Test error.
            fig = get_figure(fig_idx)
            fig_idx += 1
            to_plot = [e[1] for e in errors[cname]]
            complete(to_plot, len(x_vals))
            pyplot.plot(x_vals, to_plot, label=exp_name)

        # Evolution of lambda_t.
        fig = get_figure(fig_idx)
        fig_idx += 1
        pyplot.plot(x_vals[0:len(lambdas)], lambdas, label=exp_name)


    fig_idx = 1
    for cname in sorted(errors):
        # Offline training error.
        pyplot.figure(fig_idx)
        fig_idx += 1
        pyplot.yscale('log')
        pyplot.xlabel('n_samples')
        pyplot.ylabel('offline training %s' % cname)
        pyplot.legend()
        if expdir is not None:
            pyplot.savefig(os.path.join(expdir, 'train_%s.%s' % (cname, plot_ext)))

        # Test error.
        pyplot.figure(fig_idx)
        fig_idx += 1
        pyplot.yscale('log')
        pyplot.xlabel('n_samples')
        pyplot.ylabel('test %s' % cname)
        pyplot.legend()
        if expdir is not None:
            pyplot.savefig(os.path.join(expdir, 'test_%s.%s' % (cname, plot_ext)))

    # Lambda.
    pyplot.figure(fig_idx)
    fig_idx += 1
    pyplot.xlabel('k')
    pyplot.ylabel('lambda_t')
    pyplot.legend()
    if expdir is not None:
        pyplot.savefig(os.path.join(expdir, 'lambda_t.%s' % plot_ext))

    # Show plots.
    if show_plots:
        pyplot.show()


def test(data_spec='mnist(%(n_offline_train)s,0,%(n_test)s)', model_spec='784-%(n_hidden)s-10', n_offline_train=500, n_test=100, n_hidden=10, task='classification',
         experiments=None, show_plots=True, expdir=None, max_samples=300000):
    results = []
    model_spec = model_spec % {'n_hidden': n_hidden}
    data_spec = data_spec % {'n_offline_train': n_offline_train,
                             'n_test': n_test}
    def make_exp(spec):
        # Return dictionary of options from an experiment's spec string.
        params = spec.split('_')
        normalize = 'normalize' in params
        constrain_lambda = 'neglambda' not in params
        if 'restart' in params:
            restart_every = 1
        else:
            restart_every = 0
        batch_size = params[1]
        if batch_size == 'all':
            batch_size = n_offline_train
        else:
            batch_size = int(batch_size)
        assert batch_size <= n_offline_train
        maxiter = max_samples / batch_size
        if params[0] == 'batch':
            minibatch_size = None
            minibatch_offset = None
            n_off = batch_size
        elif params[0] == 'online':
            minibatch_size = batch_size
            minibatch_offset = int(params[2])
            n_off = n_offline_train
        else:
            raise NotImplementedError(params[0])
        return dict(
                minibatch_size=minibatch_size,
                minibatch_offset=minibatch_offset,
                maxiter=maxiter,
                normalize=normalize,
                restart_every=restart_every,
                constrain_lambda=constrain_lambda,
                n_offline_train=n_off)

    if experiments is None:
        experiments = (
            'batch_all_normalize',
            #'batch_100_normalize',
            #'batch_500_normalize',
            #'batch_1000',
            #'batch_1000_normalize',
            #'batch_1010_normalize',
            #'batch_2000',
            #'batch_2000_normalize',
            #'batch_2000_normalize_neglambda',
            #'batch_5000',
            #'batch_5000_normalize',
            #'batch_5000_normalize_neglambda',
            #'batch_10000',
            #'batch_10000_normalize',
            #'batch_10000_normalize_neglambda',
            #'batch_10000_restart',
            #'batch_50000_normalize',
            #'online_1000_1_normalize',
            #'online_1000_10_normalize',
            #'online_1000_10_normalize_neglambda',
            #'online_1000_100',
            #'online_1000_100_normalize',
            #'online_1000_100_normalize_neglambda',
            #'online_1000_1000_normalize',
            #'online_1000_1000_normalize_neglambda',
            #'online_1000_1000_normalize_restart',
            #'online_10000_1',
            #'online_10000_1_normalize',
            #'online_10000_10',
            #'online_10000_10_normalize',
            #'online_10000_100',
            #'online_10000_100_normalize',
            #'online_10000_100_normalize_neglambda',
            #'online_10000_100_normalize_restart',
            #'online_10000_1000',
            #'online_10000_1000_normalize',
            #'online_10000_1000_normalize_neglambda',
            #'online_10000_1000_restart',
            #'online_10000_10000',
            #'online_10000_10000_normalize',
            #'online_10000_10000_normalize_neglambda',
            #'online_10000_10000_normalize_restart',
            )
    else:
        experiments = experiments.split(',')

    experiments = dict((k, make_exp(k)) for k in experiments)
    for exp_name, exp_args in sorted(experiments.iteritems()):
        data_iter = get_data(data_spec)
        exp_args = exp_args.copy()
        n_off = exp_args.pop('n_offline_train')
        model = get_model(spec=model_spec, data_iter=data_iter,
                          n_offline_train=n_off,
                          n_test=n_test,
                          task=task)
        results.append([exp_name, model] + list(minimize(
                                        model=model, task=task, **exp_args)))
    plot(results, experiments, show_plots=show_plots, expdir=expdir)


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
    # Parse arguments.
    args = {}
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        args[key] = ml.util.convert_from_string(val)
        if key == 'show_plots' and not args[key]:
            matplotlib.use('pdf')
    global miniml
    assert miniml is None
    import miniml
    expdir = miniml.utility.make_expdir(state=args)
    test(expdir=expdir, **args)
    #test_ncg_2()
    return 0


if __name__ == '__main__':
    sys.exit(main())
