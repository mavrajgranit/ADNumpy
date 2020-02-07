import numpy as np
import numpy.random as random

class Operation:

    operation_name = "NOOP"

    @staticmethod
    def do(*args, **kargs):
        raise NotImplementedError("Do Operation has not been implemented.")

    @staticmethod
    def grad_left(*args, **kargs):
        raise NotImplementedError("Grad_Left Operation has not been implemented.")

    @staticmethod
    def grad_right(*args, **kargs):
        raise NotImplementedError("Grad_Right Operation has not been implemented.")

    @classmethod
    def __call__(cls, *args, **kargs):
        return cls.do(*args, **kargs)


class Add(Operation):

    operation_name = "ADD"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_t = isinstance(left, Tensor)
        r_t = isinstance(right, Tensor)
        if l_t and r_t:
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [right.value]]
            res = Tensor(left.value+right.value, left_parent=left_parent, right_parent=right_parent, operation=Add.operation_name, grad_left=Add.grad_left, grad_right=Add.grad_right)
        elif l_t and isinstance(right, (int, float)):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
            res = Tensor(left.value+right, left_parent=left_parent, right_parent=right_parent, operation=Add.operation_name, grad_left=Add.grad_left, grad_right=Add.grad_right)
        else:
            raise TypeError("Unsupported type passed for right " + Add.operation_name+": "+type(right))
        return res

    @staticmethod
    def grad_left(prev, grad):
        return prev*np.ones(grad[0].shape)

    @staticmethod
    def grad_right(prev, grad):
        return prev*np.ones(grad[0].shape)


class Sub(Operation):

    operation_name = "SUB"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_t = isinstance(left, Tensor)
        r_t = isinstance(right, Tensor)
        if l_t and r_t:
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [right.value]]
            res = Tensor(left.value-right.value, left_parent=left_parent, right_parent=right_parent, operation=Sub.operation_name, grad_left=Sub.grad_left, grad_right=Sub.grad_right)
        elif l_t and isinstance(right, (int, float, np.ndarray)):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
            res = Tensor(left.value-right, left_parent=left_parent, right_parent=right_parent, operation=Sub.operation_name, grad_left=Sub.grad_left, grad_right=Sub.grad_right)
        else:
            raise TypeError("Unsupported type passed for right "+ Sub.operation_name+": "+str(type(right)))
        return res

    @staticmethod
    def grad_left(prev, grad):
        return prev*np.ones(grad[0].shape)

    @staticmethod
    def grad_right(prev, grad):
        return prev*-np.ones(grad[0].shape)


class Mul(Operation):

    operation_name = "MUL"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_t = isinstance(left, Tensor)
        r_t = isinstance(right, Tensor)
        if l_t and r_t:
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [right.value]]
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [left.value]]
            res = Tensor(left.value*right.value, left_parent=left_parent, right_parent=right_parent, operation=Mul.operation_name, grad_left=Mul.grad_left, grad_right=Mul.grad_right)
        elif l_t and isinstance(right, (int, float)):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [right]]
            res = Tensor(left.value*right, left_parent=left_parent, right_parent=right_parent, operation=Mul.operation_name, grad_left=Mul.grad_left, grad_right=Mul.grad_right)
        else:
            raise TypeError("Unsupported type passed for right "+ Mul.operation_name+": "+type(right))
        return res

    @staticmethod
    def grad_left(prev, grad):
        return prev*grad[0]

    @staticmethod
    def grad_right(prev, grad):
        return prev*grad[0]


class Dot(Operation):

    operation_name = "DOT"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_t = isinstance(left, Tensor)
        r_t = isinstance(right, Tensor)
        if l_t and r_t:
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [right.value]]
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [left.value]]
            res = Tensor(left.value@right.value, left_parent=left_parent, right_parent=right_parent, operation=Dot.operation_name, grad_left=Dot.grad_left, grad_right=Dot.grad_right)
        else:
            raise TypeError("Unsupported type passed for right " + Dot.operation_name+": "+type(right))
        return res

    @staticmethod
    def grad_left(prev, grad):
        axes = None
        dims = grad[0].ndim
        if dims == 1:
            return prev@grad[0][np.newaxis].transpose(axes)
        elif dims > 2:
            axes = list(range(dims))
            a = axes[-2]
            axes[-2] = axes[-1]
            axes[-1] = a
        return prev@grad[0].transpose(axes)

    @staticmethod
    def grad_right(prev, grad):
        axes = None
        dims = grad[0].ndim
        if dims == 1:
            return grad[0][np.newaxis].transpose(axes)@prev
        elif dims > 2:
            axes = list(range(dims))
            a = axes[-2]
            axes[-2] = axes[-1]
            axes[-1] = a
        return grad[0].transpose(axes)@prev


class Div(Operation):

    operation_name = "DIV"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_t = isinstance(left, Tensor)
        r_t = isinstance(right, Tensor)
        if l_t and r_t:
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [right.value]]
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [left.value, right.value]]
            res = Tensor(left.value/right.value, left_parent=left_parent, right_parent=right_parent, operation=Div.operation_name, grad_left=Div.grad_left, grad_right=Div.grad_right)
        elif l_t and isinstance(right, (int, float)):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [right]]
            res = Tensor(left.value/right, left_parent=left_parent, right_parent=right_parent, operation=Div.operation_name, grad_left=Div.grad_left, grad_right=Div.grad_right)
        elif r_t and isinstance(left, (int, float)):
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [left, right.value]]
            res = Tensor(left/right.value, left_parent=left_parent, right_parent=right_parent, operation=Div.operation_name, grad_left=Div.grad_left, grad_right=Div.grad_right)
        else:
            raise TypeError("Unsupported type passed for right " + Div.operation_name+": "+type(right))
        return res

    @staticmethod
    def grad_left(prev, grad):
        return prev*1/grad[0]

    @staticmethod
    def grad_right(prev, grad):
        return prev*-grad[0]/grad[1]**2


class Pow(Operation):

    operation_name = "POW"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_t = isinstance(left, Tensor)
        r_t = isinstance(right, Tensor)
        if l_t and r_t:
            r = left.value ** right.value
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value, right.value]]
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [left.value, r]]
            res = Tensor(r, left_parent=left_parent, right_parent=right_parent, operation=Pow.operation_name, grad_left=Pow.grad_left, grad_right=Pow.grad_right)
        elif l_t and isinstance(right, (int, float)):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value, right]]
            res = Tensor(left.value**right, left_parent=left_parent, right_parent=right_parent, operation=Pow.operation_name, grad_left=Pow.grad_left, grad_right=Pow.grad_right)
        elif r_t and isinstance(left, (int, float)):
            r = left ** right.value
            if right.variable or right.right_parent or right.left_parent:
                right_parent = [right, [left, r]]
            res = Tensor(left**right.value, left_parent=left_parent, right_parent=right_parent, operation=Pow.operation_name, grad_left=Pow.grad_left, grad_right=Pow.grad_right)
        else:
            raise TypeError("Unsupported type passed for right " + Pow.operation_name+": "+type(right))
        return res

    @staticmethod
    def grad_left(prev, grad):
        return prev*grad[1]*grad[0]**(grad[1]-1)

    @staticmethod
    def grad_right(prev, grad):
        return prev*np.log(grad[0])*grad[1]


class Sum(Operation):
    operation_name = "SUM"

    @staticmethod
    def do(left, axis=0, keepdims=False):
        left_parent = None

        if isinstance(left, Tensor):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
            res = Tensor(left.value.sum(axis=axis, keepdims=keepdims), left_parent=left_parent, operation=Sum.operation_name, grad_left=Sum.grad_left)
        return res

    @staticmethod
    def grad_left(prev, grad):
        return prev * np.ones(grad[0].shape)

    @staticmethod
    def grad_right(prev, grad):
        return 0


class Transpose(Operation):
    operation_name = "TRANSPOSE"

    @staticmethod
    def do(left, axes=None):
        left_parent = None

        if isinstance(left, Tensor):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [axes]]
            if left.value.ndim == 1:
                r = left.value[np.newaxis].transpose(axes)
            else:
                r = left.value.transpose(axes)
            res = Tensor(r, left_parent=left_parent,
                         operation=Transpose.operation_name, grad_left=Transpose.grad_left)
        return res

    @staticmethod
    def grad_left(prev, grad):
        axes = grad[0]
        return prev.transpose(axes)

    @staticmethod
    def grad_right(prev, grad):
        return 0


class Reshape(Operation):
    operation_name = "RESHAPE"

    @staticmethod
    def do(left, shape):
        left_parent = None

        if isinstance(left, Tensor):
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.shape]]
            res = Tensor(left.value.reshape(shape), left_parent=left_parent,
                         operation=Reshape.operation_name, grad_left=Reshape.grad_left)
        return res

    @staticmethod
    def grad_left(prev, grad):
        shape = grad[0]
        return prev.reshape(shape)

    @staticmethod
    def grad_right(prev, grad):
        return 0


class GetItem(Operation):
    operation_name = "GETITEM"

    @staticmethod
    def do(left, item):
        left_parent = None

        if isinstance(left, Tensor):
            if isinstance(item, Tensor):
                item = item.value

            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value, item]]
            res = Tensor(left.value[item], left_parent=left_parent,
                         operation=GetItem.operation_name, grad_left=GetItem.grad_left)
        return res

    #TODO Maybe iterate over array and pass arguments in backward
    @staticmethod
    def grad_left(prev, grad):
        value = grad[0]
        item = grad[1]
        g = np.zeros(value.shape)
        g[item] = 1
        return prev*g

    @staticmethod
    def grad_right(prev, grad):
        return 0


class Argmax(Operation):
    operation_name = "ARGMAX"

    @staticmethod
    def do(left, axis=None):
        left_parent = None

        if isinstance(left, Tensor):
            idxs = left.value.argmax(axis)
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value, left.value]]
            return Tensor(idxs, left_parent=left_parent,
                          operation=Argmax.operation_name, grad_left=Argmax.grad_left)

    @staticmethod
    def grad_left(prev, grad):
        value = grad[0]
        return np.zeros(value.shape)

    @staticmethod
    def grad_right(prev, grad):
        return 0


class Take(Operation):

    operation_name = "TAKE"

    @staticmethod
    def do(left, indices, axis):
        left_parent = None

        if isinstance(left, Tensor):
            if isinstance(indices, Tensor):
                indices = indices.value
                if left.value.ndim > 1:
                    indices = indices[np.newaxis]
            elif isinstance(indices, list):
                if left.value.ndim > 1:
                    indices = [indices]
                indices = np.array(indices)

            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value, indices, axis]]

            r = np.take_along_axis(left.value, indices, axis)
            return Tensor(r, left_parent=left_parent, right_parent=None, operation=Take.operation_name,
                          grad_left=Take.grad_left, grad_right=Take.grad_right)

    @staticmethod
    def grad_left(prev, grad):
        value = grad[0]
        indices = grad[1]
        axis = grad[2]
        g = np.zeros(value.shape)
        np.put_along_axis(g, indices, 1, axis)
        return g*prev

    @staticmethod
    def grad_right(prev, grad):
        return 0


class Put(Operation):

    operation_name = "PUT"

    @staticmethod
    def do(left, indices, axis, value, inplace=False):
        left_parent = None

        if isinstance(left, Tensor):
            if isinstance(indices, Tensor):
                indices = indices.value
                if left.value.ndim > 1:
                    indices = indices[np.newaxis]
            elif isinstance(indices, list):
                if left.value.ndim > 1:
                    indices = [indices]
                indices = np.array(indices)

            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value, indices, axis]]

            if inplace:
                r = left.value
                np.put_along_axis(r, indices, value, axis)
                return left
            else:
                r = left.value.copy()

            np.put_along_axis(r, indices, value, axis)
            return Tensor(r, left_parent=left_parent, right_parent=None, operation=Put.operation_name,
                          grad_left=Put.grad_left, grad_right=Put.grad_right)

    @staticmethod
    def grad_left(prev, grad):
        return prev*0

    @staticmethod
    def grad_right(prev, grad):
        return 0


class CONV1D(Operation):

    operation_name = "CONV1D"

    @staticmethod
    def do(left, right, kernelx, kernelz, out_channels):
        left_parent = None
        right_parent = None

        if isinstance(left, Tensor) and isinstance(right, Tensor):
            conv = conv1d(right.value, kernelx, kernelz)

            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [conv]]
            if right.variable or right.left_parent or right.right_parent:
                right_parent = [right, [left.value, kernelx, kernelz, out_channels]]

            x = np.einsum('...ijk,...lkm->...ijm', left.value, conv)
            #x = left.value@conv#np.dot(left.value, conv).reshape((conv.shape[0], out_channels*conv.shape[1], left.value.shape[2], conv.shape[3]))
            return Tensor(x, left_parent=left_parent, right_parent=right_parent, operation=CONV1D.operation_name,
                          grad_left=CONV1D.grad_left, grad_right=CONV1D.grad_right)

    @staticmethod
    def grad_left(prev, grad):
        axes = list(range(grad[0].ndim))
        a = axes[-2]
        axes[-2] = axes[-1]
        axes[-1] = a
        x = np.einsum('...ijk,...lkm->...ijm', prev, grad[0].transpose((0, 1, 3, 2)))
        #x = prev@grad[0].transpose((0, 1, 3, 2))#np.dot(prev, grad[0].transpose((0, 1, 3, 2)))
        #x = x.reshape((grad[0].shape[0], grad[0].shape[1], prev.shape[2], prev.shape[3]))
        return x

    @staticmethod
    def grad_right(prev, grad):
        w = grad[0]
        kx = grad[1]
        kz = grad[2]
        out_channels = grad[3]
        conv_tr = conv1d_transpose(prev, kx, kz, out_channels)
        return w@conv_tr


def conv1d(value, kernelx, kernelz):
    if isinstance(value, Tensor):
        v = value.value
    elif isinstance(value, np.ndarray):
        v = value

    if v.ndim == 3:
        v = np.expand_dims(v, axis=0)
    elif v.ndim == 2:
        v = np.expand_dims(v, axis=0)
        v = np.expand_dims(v, axis=0)

    batch = v.shape[0]
    nx = v.shape[3] - kernelx + 1
    nz = v.shape[1] - kernelz + 1

    conv = np.zeros((batch, nz, kernelx * kernelz, nx))

    for z in range(nz):
        for x in range(nx):
            conv[:, None, z, :, None, x] = v[:, z:z + kernelz, :, x:x + kernelx].reshape((batch, 1, kernelx * kernelz, 1))

    return conv


def conv1d_transpose(value, kernelx, kernelz, out_channels):
    if isinstance(value, Tensor):
        v = value.value
    elif isinstance(value, np.ndarray):
        v = value

    if v.ndim == 3:
        v = np.expand_dims(v, axis=0)
    elif v.ndim == 2:
        v = np.expand_dims(v, axis=0)
        v = np.expand_dims(v, axis=0)

    v = np.pad(v, ((0, 0), (kernelz-1, kernelz-1), (0, 0), (kernelx-1, kernelx-1)), 'constant')
    batch = v.shape[0]
    nx = v.shape[3] - kernelx + 1
    nz = v.shape[1] - kernelz + 1 - out_channels + 1

    conv = np.zeros((batch, nz, kernelx * kernelz, nx))
    for z in range(0, nz):
        uz = z + (kernelz - 1)  # + kz-1
        lz = z - 1
        if lz < 0:
            lz = None
        for x in range(0, nx):
            ux = x + (kernelx - 1)  # + kz-1
            lx = x - 1
            if lx < 0:
                lx = None
            conv[:, None, z, :, None, x] = v[:, uz:lz:-1, :, ux:lx:-1].reshape((batch, 1, kernelx*kernelz, 1))
    return conv


class Equals(Operation):

    operation_name = "EQUALS"

    @staticmethod
    def do(left, right):
        left_parent = None
        right_parent = None

        l_v = None
        r_v = None
        if isinstance(left, Tensor):
            l_v = left.value
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
        elif isinstance(left, np.ndarray):
            l_v = left
        elif isinstance(left, list):
            l_v = np.array(left)
        else:
            raise TypeError("Invalid type")

        if isinstance(right, Tensor):
            r_v = right.value
            if right.variable or right.left_parent or right.right_parent:
                right_parent = [right, [right.value]]
        elif isinstance(right, np.ndarray):
            r_v = right
        elif isinstance(right, list):
            r_v = np.array(right)
        else:
            raise TypeError("Invalid type")

        return Tensor(l_v == r_v, left_parent=left_parent, right_parent=right_parent, operation=Equals.operation_name,
                      grad_left=Equals.grad_left, grad_right=Equals.grad_right)

    @staticmethod
    def grad_left(prev, grad):
        return prev * np.ones(grad[0].shape)

    @staticmethod
    def grad_right(prev, grad):
        return prev * np.ones(grad[0].shape)


class CustomFunction(Operation):

    operation_name = "CUST"

    @staticmethod
    def f(*args):
        raise NotImplementedError("F Operation has not been implemented.")

    @staticmethod
    def df(*args):
        raise NotImplementedError("DF Operation has not been implemented.")

    @classmethod
    def do(cls, left):
        left_parent = None

        if isinstance(left, Tensor):
            r = cls.f(left.value)
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, r]
            res = Tensor(r, left_parent=left_parent, operation=cls.operation_name, grad_left=cls.grad_left)
        elif isinstance(left, (int, float)):
            r = cls.f(left)
            res = Tensor([[r]], operation=cls.operation_name, grad_left=cls.grad_left)
        return res

    @classmethod
    def grad_left(cls, prev, grad):
        return prev * cls.df(grad[0])

    @staticmethod
    def grad_right(prev, grad):
        return 0


class Tanh(CustomFunction):

    operation_name = "TANH"

    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def df(x):
        return np.tanh(x)**2


def tanh(x):
    return Tanh.do(x)


class Sigmoid(CustomFunction):

    operation_name = "SIGMOID"

    @staticmethod
    def f(x):
        return 1.0/(1.0+np.exp(-x-np.max(x)))

    @staticmethod
    def df(x):
        return x*(1-x)


def sigmoid(x):
    return Sigmoid.do(x)


class Softmax(CustomFunction):

    operation_name = "SOFTMAX"

    @staticmethod
    def f(x):
        sx = x-np.max(x)
        exps = np.exp(sx)
        return exps/np.sum(exps, axis=x.ndim-1, keepdims=True)

    @staticmethod
    def df(x):
        #s = x.reshape(-1, 1)
        return x*(1-x)#np.diagflat(s) - np.dot(s, s.T)


class StatefulFunction(Operation):

    operation_name = "STATEF"

    def f(self, *args, **kargs):
        raise NotImplementedError("F Operation has not been implemented.")

    def df(self, *args, **kargs):
        raise NotImplementedError("DF Operation has not been implemented.")

    def do(self, left):
        left_parent = None

        if isinstance(left, Tensor):
            r = self.f(left.value)
            if left.variable or left.left_parent or left.right_parent:
                left_parent = [left, [left.value]]
            res = Tensor(r, left_parent=left_parent, operation=self.operation_name, grad_left=self.grad_left)
        elif isinstance(left, (int, float)):
            r = self.f(left)
            res = Tensor([[r]], operation=self.operation_name, grad_left=self.grad_left)
        return res

    def __call__(self, *args, **kargs):
        return self.do(*args, **kargs)

    def grad_left(self, prev, grad):
        return prev * self.df(grad[0])

    def grad_right(self, grad):
        return 0


class Relu(StatefulFunction):

    operation_name = "RELU"

    f_ = np.vectorize(lambda x: x if x > 0 else 0.0)
    df_ = np.vectorize(lambda x: 1.0 if x > 0 else 0.0)

    def f(self, x):
        if isinstance(x, np.ndarray):
            return self.f_(x)
        elif isinstance(x, (int, float)):
            return 0 if x < 0 else x

    def df(self, x):
        return self.df_(x)


#TODO Wont work as single method
class LeakyRelu(StatefulFunction):

    operation_name = "LRELU"

    def __init__(self, a):
        self.a = a

    def f(self, x):
        y = x.copy()
        if isinstance(x, np.ndarray):
            with np.nditer(y, op_flags=['readwrite']) as it:
                for i in it:
                    if i < 0:
                        i[...] *= self.a
            return y
        elif isinstance(x, (int, float)):
            return x*self.a if x < 0 else x

    def df(self, x):
        y = x.copy()
        with np.nditer(y, op_flags=['readwrite']) as it:
            for i in it:
                if i < 0:
                    i[...] = self.a
                else:
                    i[...] = 1.0
        return y


class Tensor:

    add = Add.do
    sub = Sub.do
    mul = Mul.do
    div = Div.do
    dot = Dot.do
    pow = Pow.do
    sum = Sum.do
    transpose = Transpose.do
    reshape = Reshape.do
    getitem = GetItem.do
    argmax = Argmax.do
    take = Take.do
    put = Put.do

    @property
    def shape(self):
        return self.value.shape

    def __init__(self, auto, variable=False, operation=None, grad_left=None, grad_right=None, left_parent=None, right_parent=None):
        shape = None
        value = None

        if isinstance(auto, tuple):
            shape = auto
        elif isinstance(auto, (list, np.ndarray)):
            value = auto
        elif isinstance(auto, (int, float, np.integer, np.float)):
            value = np.array([auto])
        else:
            raise TypeError()
        if shape is not None:
            self.value = np.zeros(shape)
        elif type(value) == np.ndarray:
            self.value = value
        elif type(value) == list:
            self.value = np.array(value)

        self.variable = variable
        self.gradient = None
        self.zero_grad()
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.operation = operation
        self.grad_left = grad_left
        self.grad_right = grad_right

    def zero_grad(self):
        self.gradient = 0  # np.zeros(self.value.shape)

    def __add__(self, other):
        return Tensor.add(self, other)

    #TODO this should always be inversed and the operations should handle it
    def __radd__(self, other):
        return Tensor.add(self, other)

    def __sub__(self, other):
        return Tensor.sub(self, other)

    def __rsub__(self, other):
        return Tensor.sub(self, other)

    def __mul__(self, other):
        return Tensor.mul(self, other)

    def __rmul__(self, other):
        return Tensor.mul(self, other)

    def __matmul__(self, other):
        return Tensor.dot(self, other)

    def __truediv__(self, other):
        return Tensor.div(self, other)

    def __rtruediv__(self, other):
        return Tensor.div(other, self)

    def __pow__(self, other):
        return Tensor.pow(self, other)

    def __rpow__(self, other):
        return Tensor.pow(other, self)

    def __eq__(self, other):
        return Equals.do(self, other)

    def item(self):
        if self.value.ndim > 1 and self.value.size > 1:
            raise Exception("Can't return item for multiple values")
        return self.value[0]

    def __repr__(self):
        v = self.variable
        op = self.operation is not None
        return "T_" + ("v" if v else self.operation if op else "") + str(self.value)

    def __getitem__(self, item):
        return GetItem.do(self, item)

    def fill(self, value):
        self.value.fill(value)

    def fill_f(self, func):
        self.value = func(self.value)

    def randfill(self, shape, min=-1, max=1):
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
    def random(shape, variable=False, min=-1, max=1):
        return Tensor(random.uniform(min, max, shape), variable=variable)

    def to_variable(self):
        self.variable = True

    def backward(self, prev=None):
        if self.left_parent or self.right_parent:
            if prev is None:
                prev = np.ones(self.shape)

            if self.left_parent:
                grad = self.grad_left(prev, self.left_parent[1])
                self.left_parent[0].backward(grad)
            if self.right_parent:
                grad = self.grad_right(prev, self.right_parent[1])
                self.right_parent[0].backward(grad)

        if self.variable and prev is not None:
            if prev.shape != self.value.shape:
                to_sum = []
                p_shape = prev.shape
                p_dim = prev.ndim

                v_shape = self.shape
                v_dim = self.value.ndim

                shape = [1] * p_dim
                shape[-v_dim:] = v_shape

                for i in range(p_dim):
                    if p_shape[i] > shape[i]:
                        to_sum.append(i)
                prev = prev.sum(tuple(to_sum), keepdims=True).reshape(v_shape)

                self.gradient += prev
            else:
                self.gradient += prev

# Layers
class Module:

    def __init__(self):
        pass

    def parameters(self):
        variables = self.__dict__
        parameters = []
        for key in variables:
            value = variables[key]
            if isinstance(value, Tensor):
                if value.variable:
                    parameters.append(value)
            elif isinstance(value, Module):
                parameters.append(value)
        return parameters

    def named_parameters(self):
        variables = self.__dict__
        parameters = {}
        for key in variables:
            value = variables[key]
            if isinstance(value, (Tensor, Module)):
                parameters[key] = value
        return parameters

    def forward(self, *args, **kargs):
        raise NotImplementedError("Forward has not been implemented.")

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)


class Sequential(Module):

    def __init__(self, layers):
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
        self.w = Tensor.random((inputs, outputs), variable=True, min=-0.01, max=0.01)
        self.b = None
        if bias:
            self.b = Tensor.random((1, outputs), variable=True, min=-0.01, max=0.01)

    def forward(self, x):
        o = x @ self.w
        if self.b is not None:
            o += self.b
        return o


# TODO Switch to more channels perhaps? More concurrency?
"""class Conv1d(Module):

    def __init__(self, in_channels, out_channels, kernelx, kernely, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernelx = kernelx
        self.kernely = kernely
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.kernel = Tensor((out_channels, in_channels, kernely, kernelx), variable=True)
        if bias:
            self.b = Tensor((1, 1, 1), variable=True)

    def forward(self, c):
        v = c.value
        nx = v.shape[2] - self.kernelx + 1
        ny = v.shape[1] - self.kernely + 1

        conv = np.zeros((self.out_channels, nx*ny, self.kernelx, self.kernely))
        for y in range(0, ny):
            for x in range(0, nx):
                i = y*nx + x
                conv[0, i, :, :] = v[0, y:y+self.kernely, x:x+self.kernelx].transpose()
        o = Tensor(conv)@self.kernel
        if self.b is not None:
            o += self.b
        return o"""


class ReshapeLayer(Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Conv1d(Module):

    def __init__(self, in_channels, out_channels, kernelx, kernelz, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernelx = kernelx
        self.kernelz = kernelz
        self.stride = stride
        self.padding = padding
        self.bias = bias
        size = 2/np.sqrt(kernelx * kernelz*out_channels)
        self.kernel = Tensor.random((1, out_channels, 1, kernelx * kernelz), variable=True, min=-size, max=size)
        if bias:
            self.b = Tensor.random((1, out_channels, 1, 1), variable=True)

    def forward(self, c):
        """
        conv = np.zeros((ny * nx, self.kernel.shape[0]))
        for y in range(0, round(ny)):
            for x in range(0, round(nx)):
                conv[x+y*self.kernelx, :] = v[y:y+self.kernely, x:x+self.kernelx].reshape(1, self.kernelx*self.kernely)
                
        o = Tensor(conv)@self.kernel
        if self.b is not None:
            o += self.b
        return o"""
        o = CONV1D.do(self.kernel, c, self.kernelx, self.kernelz, self.out_channels)
        if self.bias:
            o += self.b
        return o


class Conv2d(Module):

    def __init__(self, kernelx, kernely, stride=1, padding=0, bias=True):
        super().__init__()
        self.kernelx = kernelx
        self.kernely = kernely
        self.stride = stride
        self.padding = padding
        self.kernel = Tensor((kernelx*kernely, 1), variable=True)
        if bias:
            self.bias = Tensor((1, 1), variable=True)

    def forward(self, x):
        #for i in range(x.shape[2]):
        #    p = x[:, :, i].reshape(x.shape[0]*x.shape[1], 1)
        p = x.value.reshape(x.shape[1] * x.shape[2], x.shape[0])
        o = p@self.kernel
        if self.b is not None:
            o += self.b
        return o


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

    def __init__(self, module, lr=0.0):
        self.module = module
        #self.parameters = parameters
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
