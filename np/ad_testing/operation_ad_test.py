from np.ad.nn import Tensor
import numpy as np
import pytest
import math


def test_gradient_is_summed_correctly():
    a = Tensor(np.ones((1, 2, 1, 1, 2)))
    b = Tensor(np.ones((1, 2, 2, 1)))
    c = a+b

    assert np.equal(c.value, np.ones((1, 2, 1, 1, 2))+np.ones((1, 2, 2, 1))).all()
    c.backward()
    assert a.gradient.shape == a.shape and b.gradient.shape == b.shape


# Pure operations e.g. +, -, *, /, ....
def test_add_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([1.0])
    c = a+b

    assert np.equal(c.value, np.array([2.0, 3.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all() and np.equal(b.gradient, np.array([2.0])).all()


def test_sub_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([1.0])
    c = a-b

    assert np.equal(c.value, np.array([0.0, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all() and np.equal(b.gradient, np.array([-2.0])).all()


def test_mul_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([2.0])
    c = a*b

    assert np.equal(c.value, np.array([2.0, 4.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([2.0, 2.0])).all() and np.equal(b.gradient, np.array([3.0])).all()


def test_div_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0])
    b = Tensor([2.0])
    c = a/b

    assert np.equal(c.value, np.array([0.5, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([0.5, 0.5])).all() and np.equal(b.gradient, np.array([-0.75])).all()


def test_dot_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[1.0], [2.0]])
    c = a@b

    assert np.equal(c.value, np.array([5.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 2.0])).all() and np.equal(b.gradient, np.array([[1.0], [2.0]])).all()


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
    bg = 4.0*np.log(np.array([2.0]))
    assert np.equal(a.gradient, np.array([1.0, 2.0])).all() and np.equal(b.gradient, bg).all()

    a = Tensor([2.0, 3.0])
    b = Tensor([1.5, 2.0])

    c = a ** b
    res = np.array([2.0**1.5, 9.0])
    assert np.equal(c.value, res).all()

    c.backward()
    bg = res * np.log(np.array([2.0, 3.0]))
    assert np.equal(a.gradient, np.array([2.0**0.5, 3.0])).all() and np.equal(b.gradient, bg).all()


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


from np.ad.operation import Sigmoid, Tanh, Relu, LeakyRelu, Softmax

# Functions e.g. Sigmoid, Relu, ...
def test_sigmoid_operation_gradient_is_correct():
    a = Tensor([2.0])
    b = Sigmoid.forward(a)
    sig = 1/(1+math.exp(-2))

    assert np.equal(b.value, np.array([sig])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([sig*(1-sig)])).all()


def test_tanh_operation_gradient_is_correct():
    a = Tensor([2.0])
    b = Tanh.forward(a)
    tanh = np.tanh(2.0)

    assert np.equal(b.value, np.array([tanh])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1-tanh**2])).all()


def test_relu_operation_gradient_is_correct():
    a = Tensor([2.0, -2.0])
    b = Relu.forward(a)

    assert np.equal(b.value, np.array([2.0, 0.0])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 0.0])).all()


def test_leakyrelu_operation_gradient_is_correct():
    a = Tensor([2.0, -2.0])
    b = LeakyRelu.forward(a, a=0.1)

    assert np.equal(b.value, np.array([2.0, -0.2])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([1.0, 0.1])).all()


def test_softmax_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0, 3.0])
    b = Softmax.forward(a)

    assert np.isclose(b.value, np.array([0.09003057, 0.24472847, 0.66524096])).all()
    b.backward()
    assert np.equal(a.gradient, np.array([0, -2, -6])).all()


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
    assert np.equal(a.gradient, np.array([1.0])).all() and np.equal(b.gradient, np.array([1.0])).all() and np.equal(c.gradient, np.array([1.0])).all()


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
    assert np.equal(a.gradient, np.array([[1.0], [0.0]])).all() and np.equal(b.gradient, np.array([[1.0], [0.0]])).all()

pytest.main(["-x", "operation_ad_test.py", ""])
