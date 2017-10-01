u"""勾配関連の関数群"""
# coding: utf-8
import numpy

def _numerical_gradient_1d(f, x):
    u"""
    引数:
    f --
    x --

    戻り値:

    """
    h = 1e-4 # 0.0001
    grad = numpy.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す

    return grad


def numerical_gradient_2d(f, X):
    u"""
    引数:
    f --
    x --

    戻り値:

    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = numpy.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    u"""
    引数:
    f --
    x --

    戻り値:

    """
    h = 1e-4 # 0.0001
    grad = numpy.zeros_like(x)

    it = numpy.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す

    return grad
