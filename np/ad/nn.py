from np.ad.operation import *
import numpy as np
import numpy.random as random
import ctypes

mkl_rt = ctypes.CDLL('mkl_rt.dll')
mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads


class Tensor:

    @property
    def shape(self):
        return self.value.shape

    def size(self, x):
        return self.value.shape[x]

    @property
    def ndim(self):
        return self.value.ndim

    def __init__(self, value, requires_grad=True, retain_grad=None, operation=None, parents=None, context=None):
        if isinstance(value, np.ndarray):
            self.value = value
        elif isinstance(value, tuple):
            self.value = np.zeros(value)
        elif isinstance(value, list):
            self.value = np.array(value)
        elif isinstance(value, (int, float)):
            self.value = np.array([value])
        else:
            raise Exception("Type not supported: " + str(type(value)))
        self.gradient = None
        self.zero_grad()
        self.requires_grad = requires_grad
        self.retain_grad = True if (
                    retain_grad is None and operation is None) else retain_grad if retain_grad else False
        self.operation = operation
        self.parents = parents
        self.context = context

    def backward(self, grad=None, retain_grad=False):
        if self.requires_grad:
            if self.operation is not None:
                if grad is None:
                    grad = np.ones(self.value.shape)
                grads = self.operation.backward(self.context, grad)
                for i in range(len(grads)):
                    g = grads[i]
                    if g is not None:
                        self.parents[i].backward(g)

            elif grad is None:
                raise Exception("Cannot derive a boundary variable.")

            if self.retain_grad or retain_grad:
                if grad.shape != self.value.shape:
                    to_sum = []
                    p_shape = grad.shape
                    p_dim = grad.ndim

                    v_shape = self.shape
                    v_dim = self.value.ndim

                    shape = [1] * p_dim
                    shape[-v_dim:] = v_shape

                    for i in range(p_dim):
                        if p_shape[i] > shape[i]:
                            to_sum.append(i)
                    grad = grad.sum(tuple(to_sum), keepdims=True).reshape(v_shape)

                    self.gradient += grad
                else:
                    self.gradient += grad

    def zero_grad(self):
        self.gradient = np.zeros(self.value.shape)

    def item(self, *args):
        try:
            return self.value.item(*args)
        except Exception:
            raise Exception("Can't return item for multiple values")

    def __getitem__(self, item):
        return GetItem.forward(self, item=item)

    def fill_(self, value):
        self.value.fill(value)

    def fill_f_(self, func):
        self.value = func(self.value)

    def randfill_(self, shape, min=-1, max=1):
        self.value = random.uniform(min, max, shape)

    def put_(self, indices, axis, value):
        if isinstance(indices, Tensor):
            indices = indices.value
            if self.value.ndim == 1:
                indices = indices[np.newaxis]
        elif isinstance(indices, list):
            if self.value.ndim == 1:
                indices = [indices]
            indices = np.array(indices)

        np.put_along_axis(self.value, indices, value, axis)

    @staticmethod
    def random(shape, min=-1.0, max=1.0, requires_grad=False, retain_grad=None):
        return Tensor(random.uniform(min, max, shape), requires_grad=requires_grad, retain_grad=retain_grad)

    def transpose(self, axes=None):
        return Transpose.forward(self, axes=axes)

    def reshape(self, shape):
        return Reshape.forward(self, shape=shape)

    @staticmethod
    def stack(tensors, axis=0):
        return Stack.forward(tensors, axis=axis)

    def take(self, indices, axis):
        return Take.forward(self, indices=indices, axis=axis)

    def put(self, indices, axis, value, inplace=False):
        return Put.forward(self, indices=indices, axis=axis, value=value, inplace=inplace)

    def argmax(self, axis):
        return Argmax.forward(self, axis=axis)

    def sum(self, axis=0, keepdims=False):
        return Sum.forward(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=0, keepdims=False):
        return Mean.forward(self, axis=axis, keepdims=keepdims)

    def add(self, other, **kargs):
        return Add.forward(self, other, **kargs)

    def __add__(self, other, **kargs):
        return self.add(other, **kargs)

    def __radd__(self, other, **kargs):
        return Add.forward(other, self, **kargs)

    def sub(self, other, **kargs):
        return Sub.forward(self, other, **kargs)

    def __sub__(self, other, **kargs):
        return self.sub(other, **kargs)

    def __rsub__(self, other, **kargs):
        return Sub.forward(other, self, **kargs)

    def mul(self, other, **kargs):
        return Mul.forward(self, other, **kargs)

    def __mul__(self, other, **kargs):
        return self.mul(other, **kargs)

    # TODO Somehow overide Numpy's rmul
    def __rmul__(self, other, **kargs):
        return Mul.forward(other, self, **kargs)

    def div(self, other, **kargs):
        return Div.forward(self, other, **kargs)

    def __truediv__(self, other, **kargs):
        return self.div(other, **kargs)

    def __rtruediv__(self, other, **kargs):
        return Div.forward(other, self, **kargs)

    def dot(self, other, **kargs):
        return Dot.forward(self, other, **kargs)

    def __matmul__(self, other, **kargs):
        return self.dot(other, **kargs)

    def __rmatmul__(self, other, **kargs):
        return Dot.forward(other, self, **kargs)

    def powi(self, other):
        return Powi.forward(self, n=other)

    def pow(self, other):
        return Pow.forward(self, other)

    def __pow__(self, other):
        return self.pow(other)

    def equals(self, other):
        return Equals.forward(self, other)

    def __eq__(self, other):
        return self.equals(other)

    def __repr__(self):
        op = "v"
        if self.operation is not None:
            op = self.operation.__repr__()
        return "T_%s%s" % (op, self.value)


class Context:

    def __init__(self):
        self.saved_tensors = ()

    def save(self, tensors):
        self.saved_tensors = tensors


class Variable(Tensor):

    def __init__(self, tensor):
        if isinstance(tensor, Tensor):
            super().__setattr__("tensor", tensor)
        else:
            raise TypeError("Only Tensor variables are supported.")

    def __getattr__(self, name):
        result = getattr(self.tensor, name)
        return result

    def __setattr__(self, name, value):
        setattr(self.tensor, name, value)

    def __delattr__(self, name):
        delattr(self.tensor, name)

    def __repr__(self):
        return repr(self.tensor)


# Layers
class Module:

    def __init__(self):
        pass

    def parameters(self):
        variables = self.__dict__
        parameters = []
        for key in variables:
            value = variables[key]
            if isinstance(value, Variable):
                parameters.append(value.tensor)
            elif isinstance(value, Module):
                parameters += value.parameters()
        return parameters

    def named_parameters(self):
        variables = self.__dict__
        parameters = {}
        for key in variables:
            value = variables[key]
            if isinstance(value, Variable):
                parameters[key] = value.tensor
            elif isinstance(value, Module):
                parameters[key] = value
        return parameters

    def forward(self, *args, **kargs):
        raise NotImplementedError("Forward has not been implemented.")

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)


class Sequential(Module):

    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        variables = self.__dict__
        for l in layers:
            name = type(l).__name__
            var_name = name+"_0"
            while var_name in variables:
                i = int(var_name[-1])
                i += 1
                var_name = name+"_"+str(i)
            variables[var_name] = l

    def forward(self, *args, **kargs):
        x = None
        for l in self.layers:
            if x is None:
                x = l(*args, **kargs)
            else:
                x = l(x)
        return x


class Linear(Module):

    def __init__(self, inputs, outputs, bias=True):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.w = Variable(Tensor.random((inputs, outputs), min=-0.01, max=0.01, requires_grad=True))
        self.b = None
        if bias:
            self.b = Variable(Tensor.random((1, outputs), min=-0.01, max=0.01, requires_grad=True))

    def forward(self, x):
        o = x @ self.w
        if self.b is not None:
            o += self.b
        return o


class Conv1d(Module):

    def __init__(self, kx, ky, channel=1, stride=(1, 1), bias=True):
        self.kx = kx
        self.ky = ky
        self.channel = channel
        self.stride = stride
        self.k = Variable(Tensor.random((channel, ky*kx), min=-0.01, max=0.01, requires_grad=True))
        self.bias = bias
        if bias:
            self.b = Variable(Tensor.random((channel, 1), min=-0.01, max=0.01, requires_grad=True))

    def forward(self, x):
        c = conv1d.forward(x, self.k, kx=self.kx, ky=self.ky, channel=self.channel, stride=self.stride)
        if self.bias:
            c += self.b
        return c

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ReshapeLayer(Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class DynamicModule:

    def __init__(self):
        pass

    def parameters(self):
        variables = self.__dict__
        parameters = []
        for key in variables:
            value = variables[key]
            if isinstance(value, Variable):
                parameters.append(value.tensor)
            elif isinstance(value, DynamicModule):
                parameters.append(value)
        return parameters

    def named_parameters(self):
        variables = self.__dict__
        parameters = {}
        for key in variables:
            value = variables[key]
            if isinstance(value, (Tensor, DynamicModule)):
                parameters[key] = value
        return parameters

    def forward(self, *args, **kargs):
        raise NotImplementedError("Forward has not been implemented.")

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)


# Losses
class MSE_Loss:

    def __init__(self):
        pass

    def calculate(self, x, t, axis=0, keepdims=False) -> Tensor:
        return ((x-t)**2/2.0).sum(axis=axis, keepdims=keepdims)

    def __call__(self, *args, **kargs):
        return self.calculate(*args, **kargs)


# Optimizers
class SGD:

    def __init__(self, parameters, lr=0.0):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.value -= p.gradient*self.lr

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class DynamicSGD:

    def __init__(self, module, lr=0.0):
        self.module = module
        self.lr = lr

    def step(self, module=None):
        if module is not None:
            parameters = module.parameters()
        else:
            parameters = self.module.parameters()

        for p in parameters:
            if isinstance(p, Tensor):
                p.value -= p.gradient*self.lr
            elif isinstance(p, Module):
                self.step(module=p)

    def zero_grad(self, module=None):
        if module is not None:
            parameters = module.parameters()
        else:
            parameters = self.module.parameters()

        for p in parameters:
            if isinstance(p, Tensor):
                p.zero_grad()
            elif isinstance(p, Module):
                self.zero_grad(module=p)
