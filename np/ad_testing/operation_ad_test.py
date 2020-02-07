import np.ad.nn as nn
from np.ad.nn import Tensor
import numpy as np
import pytest


def test_gradient_is_summed_correctly():
    a = Tensor(np.ones((1, 2, 1, 1, 2)), variable=True)
    b = Tensor(np.ones((1, 2, 2, 1)), variable=True)
    c = a+b

    assert np.equal(c.value, np.ones((1, 2, 1, 1, 2))+np.ones((1, 2, 2, 1))).all()
    c.backward()
    assert a.gradient.shape == a.shape and b.gradient.shape == b.shape


def test_add_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([1.0], variable=True)
    c = a+b

    assert np.equal(c.value, np.array([2.0, 3.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all() and np.equal(b.gradient, np.array([2.0])).all()


def test_sub_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([1.0], variable=True)
    c = a-b

    assert np.equal(c.value, np.array([0.0, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all() and np.equal(b.gradient, np.array([-2.0])).all()


def test_mul_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([2.0], variable=True)
    c = a*b

    assert np.equal(c.value, np.array([2.0, 4.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([2.0, 2.0])).all() and np.equal(b.gradient, np.array([3.0])).all()


def test_div_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([2.0], variable=True)
    c = a/b

    assert np.equal(c.value, np.array([0.5, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([0.5, 0.5])).all() and np.equal(b.gradient, np.array([-0.75])).all()


def test_dot_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([[1.0], [2.0]], variable=True)
    c = a@b

    assert np.equal(c.value, np.array([5.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 2.0])).all() and np.equal(b.gradient, np.array([[1.0], [2.0]])).all()


def test_pow_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([2.0], variable=True)
    c = a**b
    assert np.equal(c.value, np.array([1.0, 4.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([2.0, 4.0])).all() and np.equal(b.gradient, (c.value*np.log(a.value)).sum()).all()


def test_sum_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0], [1.0, 2.0]], variable=True)
    c = a.sum((0, 1))

    assert np.equal(c.value, np.array([6.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([[1.0, 1.0], [1.0, 1.0]])).all()


def test_transpose_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    c = a.transpose()

    assert np.equal(c.value, np.array([[1.0], [2.0]])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all()


def test_reshape_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    c = a.reshape((2, 1))

    assert np.equal(c.value, np.array([[1.0], [2.0]])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all()


def test_getitem_operation_gradient_is_correct():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], variable=True)
    c = a[0]

    assert np.equal(c.value, np.array([1.0, 2.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([[1.0, 1.0], [0.0, 0.0]])).all()


def test_argmax_operation_gradient_is_correct():
    a = Tensor([[3.0, 2.0], [1.0, 4.0]], variable=True)
    c = a.argmax(0)

    assert np.equal(c.value, np.array([0.0, 1.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([[0.0, 0.0], [0.0, 0.0]])).all()


def test_take_operation_gradient_is_correct():
    a = Tensor([[3.0, 2.0], [1.0, 4.0]], variable=True)
    c = a.take(Tensor([0, 1]), 0)

    assert np.equal(c.value, np.array([3.0, 4.0])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([[1.0, 0.0], [0.0, 1.0]])).all()


def test_equals_operation_gradient_is_correct():
    a = Tensor([1.0, 2.0], variable=True)
    b = Tensor([1.0, 3.0], variable=True)
    c = a == b

    assert np.equal(c.value, np.array([True, False])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([1.0, 1.0])).all() and np.equal(b.gradient, np.array([[1.0], [1.0]])).all()


def test_conv1d_operation_gradient_is_correct():
    a = Tensor([[1, 2, 3, 4]], variable=True)
    b = Tensor([[[1, 2, 3]], [[4, 5, 6]]], variable=True)
    c = nn.CONV1D.do(a, b, 2, 2, 1)

    assert np.equal(c.value, np.array([[[[37, 47]]]])).all()
    c.backward()
    assert np.equal(a.gradient, np.array([[3, 5, 9, 11]])).all()
    assert np.equal(b.gradient, np.array([[[1, 3, 2]], [[3, 7, 4]]])).all()


pytest.main(["-x", "operation_ad_test.py", ""])
