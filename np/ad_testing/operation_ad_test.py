from np.ad.nn import Tensor
import numpy as np
import pytest
import math


def test_gradient_is_summed_correctly():
    a = Tensor(np.ones((1, 2, 1, 1, 2)))
    b = Tensor(np.ones((1, 2, 2, 1)))
    c = a + b

    assert np.equal(c.value, np.ones((1, 2, 1, 1, 2)) + np.ones((1, 2, 2, 1))).all()
    c.backward()
    assert a.gradient.shape == a.shape
    assert b.gradient.shape == b.shape


# Basic operations e.g. +, -, *, /, ....
def test_add_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([1.0])
    c = a + b

    assert np.equal(c.value, np.array([2.0, 3.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all()
    assert np.equal(b.gradient, np.array([2.0])).all()


def test_sub_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([1.0])
    c = a - b

    assert np.equal(c.value, np.array([0.0, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all()
    assert np.equal(b.gradient, np.array([-2.0])).all()


def test_mul_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([2.0])
    c = a * b

    assert np.equal(c.value, np.array([2.0, 4.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([2.0, 2.0])).all()
    assert np.equal(b.gradient, np.array([3.0])).all()


def test_div_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([2.0])
    c = a / b

    assert np.equal(c.value, np.array([0.5, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([0.5, 0.5])).all()
    assert np.equal(b.gradient, np.array([-0.75])).all()


def test_dot_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[1.0], [2.0]])
    c = a @ b

    assert np.equal(c.value, np.array([5.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 2.0])).all()
    assert np.equal(b.gradient, np.array([[1.0], [2.0]])).all()


def test_powi_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    c = a.powi(2)
    assert np.equal(c.value, np.array([1.0, 4.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([2.0, 4.0])).all()


def test_pow_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([2.0])

    c = a ** b
    assert np.equal(c.value, np.array([1.0, 4.0])).all()

    c.backward()
    bg = 4.0 * np.log(np.array([2.0]))
    assert np.equal(a.gradient, np.array([1.0, 2.0])).all()
    assert np.equal(b.gradient, bg).all()

    a = Tensor([2.0, 3.0])
    b = Tensor([1.5, 2.0])

    c = a ** b
    res = np.array([2.0 ** 1.5, 9.0])
    assert np.equal(c.value, res).all()

    c.backward()
    bg = res * np.log(np.array([2.0, 3.0]))
    assert np.equal(a.gradient, np.array([2.0 ** 0.5, 3.0])).all()
    assert np.equal(b.gradient, bg).all()


def test_sum_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = a.sum()

    assert np.equal(b.value, np.array([3.0])).all()

    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all()

    a = Tensor([[1.0, 2.0]])
    b = a.sum(axis=0, keepdims=True)

    assert np.equal(b.value, np.array([1.0, 2.0])).all()

    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all()


from np.ad.operation import Sigmoid, Tanh, Relu, leaky_relu, LeakyRelu, softmax, Softmax


# Functions e.g. Sigmoid, Relu, ...
def test_sigmoid_operation_gradient_is_correct():
    a = Tensor([2.0])
    b = Sigmoid.forward(a)
    sig = 1 / (1 + math.exp(-2))

    assert np.equal(b.value, np.array([sig])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([sig * (1 - sig)])).all()


def test_tanh_operation_gradient_is_correct():
    a = Tensor([2.0])
    b = Tanh.forward(a)
    tanh = np.tanh(2.0)

    assert np.equal(b.value, np.array([tanh])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1 - tanh ** 2])).all()


def test_relu_operation_gradient_is_correct():
    a = Tensor([2.0, -2.0])
    b = Relu.forward(a)

    assert np.equal(b.value, np.array([2.0, 0.0])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 0.0])).all()


def test_leakyrelu_operation_gradient_is_correct():
    a = Tensor([2.0, -2.0])
    b = leaky_relu.forward(a, a=0.1)

    assert np.equal(b.value, np.array([2.0, -0.2])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 0.1])).all()

    lr = LeakyRelu(0.1)
    a = Tensor([2.0, -2.0])
    b = lr(a)

    assert np.equal(b.value, np.array([2.0, -0.2])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 0.1])).all()


def test_softmax_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0, 3.0])
    b = softmax.forward(a)

    assert np.isclose(b.value, np.array([0.09003057, 0.24472847, 0.66524096])).all()
    b.backward()
    assert np.isclose(a.gradient, np.array([0.08192507, 0.18483645, 0.22269543])).all()

    a = Tensor([1.0, 2.0, 3.0])
    sm = Softmax(0)
    b = sm(a)

    assert np.isclose(b.value, np.array([0.09003057, 0.24472847, 0.66524096])).all()
    b.backward()
    assert np.isclose(a.gradient, np.array([0.08192507, 0.18483645, 0.22269543])).all()


from np.ad.operation import Stack, FlipLR


# Operations e.g. Stack, Concat, ...
def test_transpose_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0]])
    b = a.transpose()

    assert np.equal(b.value, np.array([[1.0], [2.0]])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[1.0, 1.0]])).all()

    a = Tensor([[1.0], [2.0]])
    b = a.transpose(axes=(1, 0))

    assert np.equal(b.value, np.array([[1.0, 2.0]])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[1.0], [1.0]])).all()


def test_fliplr_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0]])
    b = FlipLR.forward(a)

    assert np.equal(b.value, np.array([[2.0, 1.0]])).all()
    b = b.mul([[1.0, 2.0]])
    b.backward()
    assert np.equal(a.gradient, np.array([[2.0, 1.0]])).all()


def test_reshape_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0]])
    b = a.reshape((2, 1))

    assert np.equal(b.value, np.array([[1.0], [2.0]])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[1.0, 1.0]])).all()

    a = Tensor([[1.0], [2.0]])
    b = a.reshape((1, 2))

    assert np.equal(b.value, np.array([[1.0, 2.0]])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[1.0], [1.0]])).all()


def test_getitem_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0]])
    b = a[0]

    assert np.equal(b.value, np.array([1.0, 2.0])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[1.0, 1.0]])).all()

    a = Tensor([[1.0, 2.0]])
    b = a[0][1]

    assert np.equal(b.value, np.array([2.0])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[0.0, 1.0]])).all()


def test_stack_operation_gradient_is_correct():
    a = Tensor([1.0])
    b = Tensor([2.0])
    c = Tensor([3.0])
    d = Stack.forward([a, b, c], axis=1)

    assert np.equal(d.value, np.array([[1.0, 2.0, 3.0]])).all()
    d.backward()
    assert np.equal(a.gradient, np.array([1.0])).all()
    assert np.equal(b.gradient, np.array([1.0])).all()
    assert np.equal(
        c.gradient, np.array([1.0])).all()


def test_take_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.take([0], 1)

    assert np.equal(b.value, np.array([[1.0], [3.0]])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[1.0, 0.0], [1.0, 0.0]])).all()


def test_put_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.put([0], 1, 2.0, inplace=False)

    assert np.equal(b.value, np.array([[2.0, 2.0], [2.0, 4.0]])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([[0.0, 0.0], [0.0, 0.0]])).all()


def test_equals_operation_gradient_is_correct():
    a = Tensor([[1.0], [2.0]])
    b = Tensor([[1.0], [0.0]])
    c = a == b

    assert np.equal(c.value, np.array([[1.0], [0.0]])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([[1.0], [0.0]])).all()
    assert np.equal(b.gradient, np.array([[1.0], [0.0]])).all()


from np.ad.operation import conv1d, conv1d_transpose, conv2d, conv2d_transpose


# Advanced Operations
def test_conv1d_operation_gradient_is_correct():
    x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    k = Tensor([[1.0, 2.0, 3.0]])
    y = conv1d.forward(x, k, kx=3, ky=1, channel=1, stride=(1, 1))

    assert np.equal(y.value, np.array([[14.0, 20.0, 26.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 4.0, 10.0, 12.0, 9.0]])).all()
    assert np.equal(k.gradient, np.array(
        [[14.0, 20.0, 26.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d.forward(x, k, kx=2, ky=1, channel=1, stride=(1, 1))

    assert np.equal(y.value, np.array([[5.0, 8.0], [14.0, 17.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 4.0, 4.0], [3.0, 10.0, 8.0]])).all()
    assert np.equal(k.gradient, np.array(
        [[37.0, 47.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0]])
    k = Tensor([[1.0, 2.0],
                [3.0, 4.0]])
    y = conv1d.forward(x, k, kx=2, ky=1, channel=2, stride=(1, 1))

    assert np.equal(y.value, np.array([[5.0, 8.0], [11.0, 18.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[10.0, 28.0, 20.0]])).all()
    assert np.equal(k.gradient, np.array(
        [[5.0, 8.0], [11.0, 18.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    k = Tensor([[1.0, 2.0],
                [3.0, 4.0]])
    y = conv1d.forward(x, k, kx=2, ky=1, channel=2, stride=(1, 1))

    assert np.equal(y.value, np.array([[5.0, 8.0], [14.0, 17.0], [11., 18.], [32., 39.]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0],
               [7.0, 8.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[16.0, 42.0, 28.0], [24.0, 62.0, 40.0]])).all()
    assert np.equal(k.gradient,
                    np.array([[37.0,
                               47.0],
                              [85.0,
                               111.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    k = Tensor([[1.0, 2.0, 3.0, 4.0]])
    y = conv1d.forward(x, k, kx=2, ky=2, channel=1, stride=(1, 1))

    assert np.equal(y.value, np.array([[37.0, 47.0]])).all()
    y = y.mul([[1.0, 2.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 4.0, 4.0], [3.0, 10.0, 8.0]])).all()
    assert np.equal(k.gradient, np.array(
        [[5.0, 8.0, 14.0, 17.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    k = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0]])
    y = conv1d.forward(x, k, kx=2, ky=2, channel=2, stride=(1, 1))

    assert np.equal(y.value, np.array([[37.0, 47.0], [85.0, 111.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[16.0, 42.0, 28.0], [24.0, 62.0, 40.0]])).all()
    assert np.equal(k.gradient,
                    np.array([[5.0,
                               8.0,
                               14.0,
                               17.0],
                              [11.0,
                               18.0,
                               32.0,
                               39.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]])
    k = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0]])
    y = conv1d.forward(x, k, kx=2, ky=2, channel=2, stride=(1, 1))

    assert np.equal(y.value, np.array([[37.0, 47.0], [67.0, 77.0], [85.0, 111.0], [163.0, 189.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0],
               [7.0, 8.0]])
    y.backward()
    assert np.equal(x.gradient,
                    np.array([[26.0, 64.0, 40.0], [76.0, 184.0, 112.0], [58.0, 136.0, 80.0]])).all()
    assert np.equal(
        k.gradient, np.array([[37.0, 47.0, 67.0, 77.0], [85.0, 111.0, 163.0, 189.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    k = Tensor([[1.0, 2.0, 3.0]])
    y = conv1d.forward(x, k, kx=3, ky=1, channel=1, stride=(2, 1))

    assert np.equal(y.value, np.array([[14.0, 26.0]])).all()
    y = y.mul([[1.0, 2.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 2.0, 5.0, 4.0, 6.0]])).all()
    assert np.equal(k.gradient, np.array(
        [[7.0, 10.0, 13.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d.forward(x, k, kx=2, ky=1, channel=1, stride=(2, 2))

    assert np.equal(y.value, np.array([[5.0, 11.0], [29.0, 35.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 2.0, 2.0, 4.0],
                                          [0.0, 0.0, 0.0, 0.0],
                                          [3.0, 6.0, 4.0, 8.0]])).all()
    assert np.equal(k.gradient,
                    np.array([[78.0, 88.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]])
    k = Tensor([[1.0, 2.0],
                [3.0, 4.0]])
    y = conv1d.forward(x, k, kx=2, ky=1, channel=2, stride=(2, 2))

    assert np.equal(y.value, np.array([[5.0, 11.0], [29.0, 35.0],
                                       [11.0, 25.0], [67.0, 81.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0],
               [7.0, 8.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[16.0, 22.0, 20.0, 28.0],
                                          [0.0, 0.0, 0.0, 0.0],
                                          [24.0, 34.0, 28.0, 40.0]])).all()
    assert np.equal(k.gradient,
                    np.array([[78.0, 88.0],
                              [174.0,
                               200.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0]])
    k = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]])
    y = conv1d.forward(x, k, kx=3, ky=3, channel=2, stride=(2, 2))

    assert np.equal(y.value, np.array([[411.0, 501.0],
                                       [861.0, 951.0],
                                       [978.0, 1230.0],
                                       [2238.0, 2490.0]])).all()
    y = y.mul([[1.0, 2.0],
               [3.0, 4.0],
               [5.0, 6.0],
               [7.0, 8.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[51.0, 57.0, 125.0, 70.0, 78.0],
                                          [69.0, 75.0, 167.0, 94.0, 102.0],
                                          [160.0, 176.0, 386.0, 214.0, 234.0],
                                          [103.0, 113.0, 243.0, 132.0, 144.0],
                                          [133.0, 143.0, 309.0, 168.0, 180.0]])).all()
    assert np.equal(k.gradient, np.array([[92.0, 102.0, 112.0, 142.0, 152.0, 162.0, 192.0, 202.0, 212.0],
                                          [204.0, 230.0, 256.0, 334.0, 360.0, 386.0, 464.0, 490.0, 516.0]])).all()


def test_conv1dtranspose_operation_gradient_is_correct():
    x = Tensor([[1.0, 2.0, 3.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=1, stride=(1, 1))

    assert np.equal(y.value, np.array([[2.0, 5.0, 8.0, 3.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 7.0, 10.0]])).all()
    assert np.equal(k.gradient, np.array([[20.0, 14.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=1, stride=(1, 1))

    assert np.equal(y.value, np.array([[2.0, 5.0, 8.0, 3.0], [8.0, 14.0, 17.0, 6.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 7.0, 10.0], [16.0, 19.0, 22.0]])).all()
    assert np.equal(k.gradient, np.array([[127.0, 106.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
    k = Tensor([[1.0, 2.0, 3.0, 4.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=2, channel=1, stride=(1, 1))

    assert np.equal(y.value, np.array([[4.0, 11.0, 18.0, 9.0], [18.0, 37.0, 47.0, 21.0], [8.0, 14.0, 17.0, 6.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[26.0, 36.0, 46.0], [66.0, 76.0, 86.0]])).all()
    assert np.equal(k.gradient, np.array([[211.0, 190.0, 127.0, 106.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=1, stride=(3, 1))

    assert np.equal(y.value, np.array([[2.0, 1.0, 0.0, 4.0, 2.0, 0.0, 6.0, 3.0, 0.0, 8.0, 4.0, 0.0, 10.0, 5.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 13.0, 22.0, 31.0, 40.0]])).all()
    assert np.equal(k.gradient, np.array([[150.0, 135.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=1, stride=(1, 2))

    assert np.equal(y.value, np.array([[2.0, 5.0, 8.0, 11.0, 4.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0],
                                       [10.0, 17.0, 20.0, 23.0, 8.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0],
                                       [18.0, 29.0, 32.0, 35.0, 12.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0, 5.0],
               [6.0, 7.0, 8.0, 9.0, 10.0],
               [11.0, 12.0, 13.0, 14.0, 15.0],
               [16.0, 17.0, 18.0, 19.0, 20.0],
               [21.0, 22.0, 23.0, 24.0, 25.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 7.0, 10.0, 13.0],
                                          [34.0, 37.0, 40.0, 43.0],
                                          [64.0, 67.0, 70.0, 73.0]])).all()
    assert np.equal(k.gradient, np.array([[1388.0, 1310.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]])
    k = Tensor([[1.0, 2.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=1, stride=(2, 2))

    assert np.equal(y.value, np.array([[2.0, 1.0, 4.0, 2.0, 6.0, 3.0, 8.0, 4.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [10.0, 5.0, 12.0, 6.0, 14.0, 7.0, 16.0, 8.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [18.0, 9.0, 20.0, 10.0, 22.0, 11.0, 24.0, 12.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
               [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
               [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
               [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 10.0, 16.0, 22.0],
                                          [52.0, 58.0, 64.0, 70.0],
                                          [100.0, 106.0, 112.0, 118.0]])).all()
    assert np.equal(k.gradient, np.array([[2180.0, 2102.0]])).all()

    x = Tensor([[1.0, 2.0, 3.0]])
    k = Tensor([[1.0, 2.0],
                [3.0, 4.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=2, stride=(1, 1))

    assert np.equal(y.value, np.array([[2.0, 5.0, 8.0, 3.0],
                                       [4.0, 11.0, 18.0, 9.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[68.0, 88.0, 108.0]])).all()
    assert np.equal(k.gradient, np.array([[20.0, 14.0], [44.0, 38.0]])).all()


def test_conv2d_operation_gradient_is_correct():
    ####################################################################################################################
    cx = 4
    cy = 1
    cz = 0
    batch = 0
    channel = 1
    kx, ky, kz = 2, 1, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[5.0, 8.0, 11.0]]])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 4.0, 7.0, 6.0]])).all()
    assert np.equal(k.gradient, np.array([[14.0, 20.0]])).all()

    ####################################################################################################################
    cx = 4
    cy = 4
    cz = 0
    batch = 0
    channel = 1
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[44.0, 54.0, 64.0],
                                        [84.0, 94.0, 104.0],
                                        [124.0, 134.0, 144.0]]])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 4.0, 7.0, 6.0],
                                          [7.0, 23.0, 33.0, 24.0],
                                          [19.0, 53.0, 63.0, 42.0],
                                          [21.0, 52.0, 59.0, 36.0]])).all()
    assert np.equal(k.gradient, np.array([[348.0, 393.0, 528.0, 573.0]])).all()

    ####################################################################################################################
    cx = 4
    cy = 4
    cz = 0
    batch = 0
    channel = 1
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 2, 2, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[44.0, 64.0],
                                        [124.0, 144.0]]])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[1.0, 2.0, 2.0, 4.0],
                                          [3.0, 4.0, 6.0, 8.0],
                                          [3.0, 6.0, 4.0, 8.0],
                                          [9.0, 12.0, 12.0, 16.0]])).all()
    assert np.equal(k.gradient, np.array([[78.0, 88.0, 118.0, 128.0]])).all()

    ####################################################################################################################
    cx = 4
    cy = 2
    cz = 0
    batch = 0
    channel = 2
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 2, 2, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[44.0, 64.0]],
                                       [[100.0, 152.0]]])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[6.0, 8.0, 12.0, 16.0],
                                          [10.0, 12.0, 20.0, 24.0]])).all()
    assert np.equal(k.gradient, np.array([[7.0, 10.0, 19.0, 22.0],
                                          [7.0, 10.0, 19.0, 22.0]])).all()

    ####################################################################################################################
    cx = 4
    cy = 2
    cz = 1
    batch = 0
    channel = 2
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 2, 2, 1
    x = Tensor(np.arange(1, cz * cy * cx + 1).reshape(cz, cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[44.0, 64.0]],
                                       [[100.0, 152.0]]])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, channel * nz * ny * nx + 1).reshape(channel * nz, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[16.0, 20.0, 22.0, 28.0],
                                          [24.0, 28.0, 34.0, 40.0]])).all()
    assert np.equal(k.gradient, np.array([[7.0, 10.0, 19.0, 22.0],
                                          [15.0, 22.0, 43.0, 50.0]])).all()

    ####################################################################################################################
    cx = 4
    cy = 4
    cz = 4
    batch = 0
    channel = 2
    kx, ky, kz = 2, 2, 2
    sx, sy, sz = 2, 2, 2
    x = Tensor(np.arange(1, cz * cy * cx + 1).reshape(cz, cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[560.0, 632.0],
                                        [848.0, 920.0]],
                                       [[1712.0, 1784.0],
                                        [2000.0, 2072.0]],
                                       [[1296.0, 1496.0],
                                        [2096.0, 2296.0]],
                                       [[4496.0, 4696.0],
                                        [5296.0, 5496.0]]])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, channel * nz * ny * nx + 1).reshape(nz * channel, ny, nx))
    y.backward()

    # assert np.equal(x.gradient, np.array().all()
    assert np.equal(k.gradient, np.array([[1084., 1120., 1228., 1264., 1660., 1696., 1804., 1840.],
                                          [2492., 2592., 2892., 2992., 4092., 4192., 4492., 4592.]])).all()

    ####################################################################################################################
    cx = 2
    cy = 2
    cz = 1
    batch = 2
    channel = 2
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, batch * cz * cy * cx + 1).reshape(batch, cz, cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([
        [[[30.0]],
         [[70.0]]],
        [[[70.0]],
         [[174.0]]]
    ])).all()
    nx, ny, nz = conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, batch * channel * nz * ny * nx + 1).reshape(batch, nz * channel, ny, nx))
    y.backward()

    assert np.equal(x.gradient, np.array([
        [[[11.0, 14.0],
          [17.0, 20.0]]],
        [[[23.0, 30.0],
          [37.0, 44.0]]]
    ])).all()
    assert np.equal(k.gradient, np.array([[16., 20., 24., 28.],
                                          [22., 28., 34., 40.]])).all()


def conv_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz):
    nx = int((cx - kx) / sx) + 1
    ny = int((cy - ky) / sy) + 1
    nz = int((cz - kz) / sz) + 1
    return nx, ny, nz


def test_conv2dtranspose_operation_gradient_is_correct():
    ####################################################################################################################
    cx = 2
    cy = 1
    cz = 0
    batch = 0
    channel = 1
    kx, ky, kz = 2, 1, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[2.0, 5.0, 2.0]]])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 7.0]])).all()
    assert np.equal(k.gradient, np.array([[8.0, 5.0]])).all()

    ####################################################################################################################
    cx = 2
    cy = 2
    cz = 0
    batch = 0
    channel = 1
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[4.0, 11.0, 6.0],
                                        [14.0, 30.0, 14.0],
                                        [6.0, 11.0, 4.0]]])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[23.0, 33.0],
                                          [53.0, 63.0]])).all()
    assert np.equal(k.gradient, np.array([[77.0, 67.0, 47.0, 37.0]])).all()

    ####################################################################################################################
    cx = 2
    cy = 2
    cz = 0
    batch = 0
    channel = 1
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 2, 2, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[4.0, 3.0, 8.0, 6.0],
                                        [2.0, 1.0, 4.0, 2.0],
                                        [12.0, 9.0, 16.0, 12.0],
                                        [6.0, 3.0, 8.0, 4.0]]])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, ny * nx + 1).reshape(1, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[26.0, 46.0],
                                          [106.0, 126.0]])).all()
    assert np.equal(k.gradient, np.array([128.0, 118.0, 88.0, 78.0])).all()

    ####################################################################################################################
    cx = 2
    cy = 1
    cz = 0
    batch = 0
    channel = 2
    kx, ky, kz = 2, 1, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, cy * cx + 1).reshape(cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([
        [[2.0, 5.0, 2.0]],
        [[4.0, 11.0, 6.0]]
    ])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, channel * ny * nx + 1).reshape(channel, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[58.0, 78.0]])).all()
    assert np.equal(k.gradient, np.array([[8.0, 5.0],
                                          [17.0, 14.0]])).all()

    ####################################################################################################################
    cx = 2
    cy = 2
    cz = 1
    batch = 0
    channel = 2
    kx, ky, kz = 2, 2, 1
    sx, sy, sz = 2, 2, 1
    x = Tensor(np.arange(1, cz * cy * cx + 1).reshape(cz, cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([
        [[4.0, 3.0, 8.0, 6.0],
         [2.0, 1.0, 4.0, 2.0],
         [12.0, 9.0, 16.0, 12.0],
         [6.0, 3.0, 8.0, 4.0]],
        [[8.0, 7.0, 16.0, 14.0],
         [6.0, 5.0, 12.0, 10.0],
         [24.0, 21.0, 32.0, 28.0],
         [18.0, 15.0, 24.0, 20.0]]
    ])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, channel * nz * ny * nx + 1).reshape(channel * nz, ny, nx))
    y.backward()
    assert np.equal(x.gradient, np.array([[792.0, 936.0],
                                          [1368.0, 1512.0]])).all()
    assert np.equal(k.gradient, np.array([[128.0, 118.0, 88.0, 78.0],
                                          [288.0, 278.0, 248.0, 238.0]])).all()

    ####################################################################################################################
    cx = 2
    cy = 2
    cz = 2
    batch = 0
    channel = 2
    kx, ky, kz = 2, 2, 2
    sx, sy, sz = 2, 2, 2
    x = Tensor(np.arange(1, cz * cy * cx + 1).reshape(cz, cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([[[8.0, 7.0, 16.0, 14.0],
                                        [6.0, 5.0, 12.0, 10.0],
                                        [24.0, 21.0, 32.0, 28.0],
                                        [18.0, 15.0, 24.0, 20.0]],
                                       [[4.0, 3.0, 8.0, 6.0],
                                        [2.0, 1.0, 4.0, 2.0],
                                        [12.0, 9.0, 16.0, 12.0],
                                        [6.0, 3.0, 8.0, 4.0]],
                                       [[40.0, 35.0, 48.0, 42.0],
                                        [30.0, 25.0, 36.0, 30.0],
                                        [56.0, 49.0, 64.0, 56.0],
                                        [42.0, 35.0, 48.0, 40.0]],
                                       [[20.0, 15.0, 24.0, 18.0],
                                        [10.0, 5.0, 12.0, 6.0],
                                        [28.0, 21.0, 32.0, 24.0],
                                        [14.0, 7.0, 16.0, 8.0]],
                                       [[16.0, 15.0, 32.0, 30.0],
                                        [14.0, 13.0, 28.0, 26.0],
                                        [48.0, 45.0, 64.0, 60.0],
                                        [42.0, 39.0, 56.0, 52.0]],
                                       [[12.0, 11.0, 24.0, 22.0],
                                        [10.0, 9.0, 20.0, 18.0],
                                        [36.0, 33.0, 48.0, 44.0],
                                        [30.0, 27.0, 40.0, 36.0]],
                                       [[80.0, 75.0, 96.0, 90.0],
                                        [70.0, 65.0, 84.0, 78.0],
                                        [112.0, 105.0, 128.0, 120.0],
                                        [98.0, 91.0, 112.0, 104.0]],
                                       [[60.0, 55.0, 72.0, 66.0],
                                        [50.0, 45.0, 60.0, 54.0],
                                        [84.0, 77.0, 96.0, 88.0],
                                        [70.0, 63.0, 80.0, 72.0]]
                                       ])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, channel * nz * ny * nx + 1).reshape(nz * channel, ny, nx))
    y.backward()

    """assert np.equal(x.gradient, np.array([
        [[8.0, 7.0],
         [6.0, 5.0]],
        [[4.0, 3.0],
         [2.0, 1.0]]
    ])).all()"""
    assert np.equal(k.gradient, np.array([[1840., 1804., 1696., 1660., 1264., 1228., 1120., 1084.],
                                          [4144., 4108., 4000., 3964., 3568., 3532., 3424., 3388.]])).all()

    ####################################################################################################################
    cx = 1
    cy = 1
    cz = 2
    batch = 2
    channel = 2
    kx, ky, kz = 1, 1, 1
    sx, sy, sz = 1, 1, 1
    x = Tensor(np.arange(1, batch * cz * cy * cx + 1).reshape(batch, cz, cy, cx))
    k = Tensor(np.arange(1, channel * kz * ky * kx + 1).reshape(channel, kz * ky * kx))
    y = conv2d_transpose.forward(x, k, kx=kx, ky=ky, kz=kz, channel=channel, stride=(sx, sy, sz))

    assert np.equal(y.value, np.array([
        [
            [[1.0]],
            [[2.0]],
            [[2.0]],
            [[4.0]]
        ],
        [
            [[3.0]],
            [[4.0]],
            [[6.0]],
            [[8.0]]
        ]
    ])).all()
    nx, ny, nz = conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz)
    y = y.mul(np.arange(1, batch*channel * nz * ny * nx + 1).reshape(batch, nz*channel, ny, nx))
    y.backward()

    assert np.equal(x.gradient, np.array([
        [[[12.0]],
         [[18.0]]],
        [[[36.0]],
         [[42.0]]]
    ])).all()
    assert np.equal(k.gradient, np.array([[44.0],
                                          [64.0]])).all()


def conv_t_shape(cx, cy, cz, kx, ky, kz, sx, sy, sz):
    tx = cx + (kx - 1) * 2 + (sx - 1) * (cx - 1)
    nx = tx - kx + 1
    ty = cy + (ky - 1) * 2 + (sy - 1) * (cy - 1)
    ny = ty - ky + 1
    tz = cz + (kz - 1) * 2 + (sz - 1) * (cz - 1)
    nz = tz - kz + 1
    return nx, ny, nz


pytest.main(["-x", "operation_ad_test.py"])
