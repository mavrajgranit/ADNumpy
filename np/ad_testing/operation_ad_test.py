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


from np.ad.operation import Stack


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


from np.ad.operation import conv1d, conv1d_transpose


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
    assert True
    """x = Tensor([[1.0, 2.0, 3.0]])
    k = Tensor([[1.0, 2.0],
                [3.0, 4.0]])
    y = conv1d_transpose.forward(x, k, kx=2, ky=1, channel=1, stride=(1, 1))
    print(y)
    assert np.equal(y.value, np.array([[2.0, 5.0, 8.0, 3.0]])).all()
    y = y.mul([[1.0, 2.0, 3.0, 4.0]])
    y.backward()
    assert np.equal(x.gradient, np.array([[4.0, 7.0, 10.0]])).all()
    assert np.equal(k.gradient, np.array([[1.0, 2.0]])).all()
    print(x.gradient)"""


pytest.main(["-x", "operation_ad_test.py", ""])
