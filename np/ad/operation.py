
class Operation:
    name = "NOOP"

    @classmethod
    def forward(cls, *args, **kargs):
        from np.ad.nn import Tensor, Context
        if isinstance(args[0], (tuple, list)):
            parents = list(args[0])
            args = [parents]
        else:
            parents = list(args)
            args = parents

        rg = False
        for i in range(len(parents)):
            p = parents[i]
            if not isinstance(p, Tensor):
                parents[i] = p = Tensor(p, requires_grad=False, retain_grad=False)
            rg = rg or p.requires_grad

        context = Context()
        result = cls.forward_(context, *args, **kargs)
        return Tensor(result, operation=cls, parents=parents, context=context, requires_grad=rg)

    @staticmethod
    def forward_(ctxt, *args, **kargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctxt, *args):
        raise Exception("Not implemented")

    @classmethod
    def __repr__(cls):
        return cls.name

    @classmethod
    def __str__(cls):
        return cls.name

    @classmethod
    def __call__(cls, *args, **kargs):
        return cls.forward(*args, **kargs)


import numpy as np


class Add(Operation):
    name = "ADD"

    @staticmethod
    def forward_(ctxt, a, b):
        grads = []

        ag = a.requires_grad
        bg = b.requires_grad

        grads.append(a.value if ag else None)
        grads.append(b.value if bg else None)

        ctxt.save(grads)
        return a.value + b.value

    @staticmethod
    def backward(ctxt, grad):
        a, b = ctxt.saved_tensors

        da = np.ones(a.shape) * grad if a is not None else None
        db = np.ones(b.shape) * grad if b is not None else None

        return [da, db]


class Sub(Operation):
    name = "SUB"

    @staticmethod
    def forward_(ctxt, a, b):
        grads = []

        ag = a.requires_grad
        bg = b.requires_grad

        grads.append(a.value if ag else None)
        grads.append(b.value if bg else None)

        ctxt.save(grads)
        return a.value - b.value

    @staticmethod
    def backward(ctxt, grad):
        a, b = ctxt.saved_tensors

        da = np.ones(a.shape) * grad if a is not None else None
        db = -np.ones(b.shape) * grad if b is not None else None

        return [da, db]


class Mul(Operation):
    name = "MUL"

    @staticmethod
    def forward_(ctxt, a, b):
        grads = []

        ag = a.requires_grad
        bg = b.requires_grad

        grads.append(b.value if ag else None)
        grads.append(a.value if bg else None)

        ctxt.save(grads)
        return a.value * b.value

    @staticmethod
    def backward(ctxt, grad):
        b, a = ctxt.saved_tensors

        da = b * grad if b is not None else None
        db = a * grad if a is not None else None

        return [da, db]


class Div(Operation):
    name = "DIV"

    @staticmethod
    def forward_(ctxt, a, b):
        grads = []

        ag = a.requires_grad
        bg = b.requires_grad

        grads.append(b.value if ag else None)
        grads.append(a.value if bg else None)
        grads.append(b.value if bg else None)

        ctxt.save(grads)
        return a.value / b.value

    @staticmethod
    def backward(ctxt, grad):
        b, a, b_ = ctxt.saved_tensors

        da = 1 / b * grad if b is not None else None
        db = -a / (b_ * b_) * grad if a is not None else None

        return [da, db]


class Dot(Operation):
    name = "DOT"

    @staticmethod
    def forward_(ctxt, a, b):
        grads = []

        ag = a.requires_grad
        bg = b.requires_grad

        grads.append(b.value if ag else None)
        grads.append(a.value if bg else None)

        ctxt.save(grads)
        return a.value @ b.value

    @staticmethod
    def backward(ctxt, grad):
        b, a = ctxt.saved_tensors

        da = grad @ b.T if b is not None else None
        db = a.T @ grad if a is not None else None

        return [da, db]


class Powi(Operation):
    name = "POWi"

    @staticmethod
    def forward_(ctxt, x, n=2):
        xg = x.requires_grad

        res = x.value
        grad = 1
        for i in range(1, n):
            if i == n - 1:
                grad = res
            res = res * x.value

        grads = (grad, n) if xg else None
        ctxt.save(grads)
        return res

    @staticmethod
    def backward(ctxt, grad):
        g = ctxt.saved_tensors
        dx = None

        if g is not None:
            grad, n = g
            dx = n * grad

        return [dx]


class Pow(Operation):
    operation_name = "POW"

    @staticmethod
    def forward_(ctxt, x, y):
        xg = x.requires_grad
        yg = y.requires_grad
        grads = []

        res = x.value**y.value

        grads.append((x.value, y.value) if xg else None)
        grads.append((res, x.value) if yg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, yg = ctxt.saved_tensors

        dx = None
        if xg is not None:
            x, y = xg
            dx = x**(y-1)

        dy = None
        if yg is not None:
            res, y = yg
            dy = res * np.log(y)

        return [dx, dy]


class Sum(Operation):
    name = "SUM"

    @staticmethod
    def forward_(ctxt, x, axis=0, keepdims=False):
        grads = []
        xg = x.requires_grad

        res = x.value.sum(axis=axis, keepdims=keepdims)

        grads.append(x.value.shape if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        shape, = ctxt.saved_tensors
        dx = grad * np.ones(shape) if shape is not None else None
        return [dx]


class Mean(Operation):
    name = "MEAN"

    @staticmethod
    def forward_(ctxt, x, axis=0, keepdims=False):
        grads = []
        xg = x.requires_grad

        res = x.value.mean(axis=axis, keepdims=keepdims)

        grads.append((x.value.shape, axis) if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        g, = ctxt.saved_tensors

        dx = None
        if g is not None:
            shape, axis = g
            # TODO multiple axis sum mean
            dx = grad * np.ones(shape)/shape[axis]
        return [dx]


class Sigmoid(Operation):
    name = "SIGMOID"

    @staticmethod
    def forward_(ctxt, x):
        grads = []
        xg = x.requires_grad

        res = 1.0 / (1 + np.exp(-x.value))  # 1.0 / (1.0 + np.exp(-x - np.max(x)))

        grads.append(res if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        y, = ctxt.saved_tensors
        dx = grad * y * (1.0 - y) if y is not None else None
        return [dx]


class Tanh(Operation):
    name = "Tanh"

    @staticmethod
    def forward_(ctxt, x):
        grads = []
        xg = x.requires_grad

        res = np.tanh(x.value)

        grads.append(res if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, = ctxt.saved_tensors
        dx = grad * (1 - xg**2) if xg is not None else None
        return [dx]


class Relu(Operation):
    name = "RELU"

    f_ = np.vectorize(lambda x: x if x > 0 else 0.0)
    df_ = np.vectorize(lambda x: 1.0 if x > 0 else 0.0)

    @staticmethod
    def forward_(ctxt, x):
        grads = []
        xg = x.requires_grad

        res = Relu.f_(x.value)

        grads.append(x.value if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, = ctxt.saved_tensors
        dx = grad * Relu.df_(xg) if xg is not None else None
        return [dx]


class LeakyRelu(Operation):
    name = "LeakyRELU"

    def __init__(self, a):
        self.a = a

    @staticmethod
    def forward_(ctxt, x, a=0.2):
        grads = []
        xg = x.requires_grad
        xv = x.value

        res = np.where(xv < 0, xv*a, xv)

        grads.append((x.value, a) if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        g, = ctxt.saved_tensors

        dx = None
        if g is not None:
            xg, a = g
            dx = grad*np.where(xg <= 0, a, 1.0)
        return [dx]


class Softmax(Operation):
    name = "SOFTMAX"

    @staticmethod
    def forward_(ctxt, x):
        grads = []
        xg = x.requires_grad

        xv = x.value
        sx = xv - np.max(xv)
        exps = np.exp(sx)
        res = exps / np.sum(exps, axis=xv.ndim - 1, keepdims=True)

        grads.append(res if xg else None)
        ctxt.save(grads)
        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, = ctxt.saved_tensors
        dx = grad * (xg * (1 - xg)) if xg is not None else None
        return [dx]


class Transpose(Operation):
    name = "TRANSPOSE"

    @staticmethod
    def forward_(ctxt, x, axes=None):
        grads = []
        xg = x.requires_grad

        xv = x.value
        if xv.ndim == 1:
            res = xv[np.newaxis].transpose(axes)
        else:
            res = xv.transpose(axes)
        grads.append(xg if xg else None)
        grads.append(axes if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, axes = ctxt.saved_tensors

        dx = grad.transpose(axes) if xg else None
        return [dx]


class Reshape(Operation):
    name = "RESHAPE"

    @staticmethod
    def forward_(ctxt, x, shape=None):
        grads = []
        xg = x.requires_grad

        res = x.value.reshape(shape)
        grads.append(x.value.shape if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        shape, = ctxt.saved_tensors

        dx = grad.reshape(shape) if shape is not None else None
        return [dx]


class GetItem(Operation):
    name = "GETITEM"

    @staticmethod
    def forward_(ctxt, x, item=None):
        grads = []
        xg = x.requires_grad

        res = x.value[item]
        grads.append((x.value, item) if xg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, = ctxt.saved_tensors

        dx = None
        if xg is not None:
            value, item = xg
            dx = np.zeros_like(value)
            dx[item] = 1.0
            dx *= grad
        return [dx]


class Stack(Operation):
    name = "STACK"

    @staticmethod
    def forward_(ctxt, tensors, axis=0):
        grads = []
        values = []
        slices = []

        lv = 0
        for i in range(len(tensors)):
            t = tensors[i]
            g = t.requires_grad
            v = t.value
            values.append(v)
            grads.append(g)
            if i > 0:
                slices.append(v.shape[axis - 1] + lv)
                lv = slices[-1]

        ctxt.save((grads, slices, axis))
        return np.stack(values, axis=axis)

    @staticmethod
    def backward(ctxt, grad):
        grads, slices, axis = ctxt.saved_tensors

        splits = np.split(grad, slices, axis=axis)

        for i in range(len(grads)):
            g = grads[i]
            splits[i] = splits[i] if g else None

        return splits


class Argmax(Operation):
    name = "ARGMAX"

    @staticmethod
    def forward_(ctxt, x, axis=None):
        grads = []
        xg = x.requires_grad

        idxs = x.value.argmax(axis)
        grads.append(x.value.shape if xg else None)

        return idxs

    @staticmethod
    def backward(ctxt, grad):
        shape, = ctxt.saved_tensors
        dx = None
        if shape is not None:
            dx = None  # np.zeros(shape)
        return [dx]


class Take(Operation):
    name = "TAKE"

    @staticmethod
    def forward_(ctxt, x, indices=None, axis=None):
        from np.ad.nn import Tensor
        grads = []
        xg = x.requires_grad

        if isinstance(indices, Tensor):
            indices = indices.value
            if x.value.ndim > 1:
                indices = indices[np.newaxis]
        elif isinstance(indices, list):
            if x.value.ndim > 1:
                indices = [indices]
            indices = np.array(indices)

        if xg:
            g = (x.value, indices, axis)
            grads.append(g)
        else:
            grads.append(None)
        ctxt.save(grads)

        res = np.take_along_axis(x.value, indices, axis)
        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, = ctxt.saved_tensors

        dx = None
        if xg is not None:
            value, indices, axis = xg
            dx = np.zeros(value.shape)
            np.put_along_axis(dx, indices, 1, axis)
            dx *= grad
        return [dx]


class Put(Operation):
    name = "PUT"

    @staticmethod
    def forward_(ctxt, x, indices=None, axis=None, value=None, inplace=False):
        from np.ad.nn import Tensor
        grads = []
        xg = x.requires_grad

        if isinstance(indices, Tensor):
            indices = indices.value
            if x.value.ndim > 1:
                indices = indices[np.newaxis]
        elif isinstance(indices, list):
            if x.value.ndim > 1:
                indices = [indices]
            indices = np.array(indices)

        if inplace:
            res = x.value
            np.put_along_axis(x, indices, value, axis)
            return x
        else:
            res = x.value.copy()

        np.put_along_axis(res, indices, value, axis)
        return res

    @staticmethod
    def backward(ctxt, grad):
        return [None]


class Equals(Operation):
    name = "EQUALS"

    @staticmethod
    def forward_(ctxt, x, y):
        grads = []
        xg = x.requires_grad
        yg = y.requires_grad

        res = np.equal(x.value, y.value).astype(float)
        grads.append(res if xg else None)
        grads.append(res if yg else None)
        ctxt.save(grads)

        return res

    @staticmethod
    def backward(ctxt, grad):
        xg, yg = ctxt.saved_tensors

        dx = None
        if xg is not None:
            dx = grad * xg

        dy = None
        if yg is not None:
            dy = grad * yg

        return [dx, dy]

