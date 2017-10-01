
u"""機械学習用汎用関数"""
# coding: utf-8
import numpy

def sigmoid(np_x):
    u"""
    シグモイド関数
    """
    return 1 / (1 + numpy.exp(-np_x))


def relu(np_x):
    u"""
    ReLU関数
    """
    return numpy.maximum(0, np_x)


def softmax(np_x):
    u"""
    ソフトマックス関数
    """
    if np_x.ndim == 2:
        np_x = np_x.T
        np_x = np_x - numpy.max(np_x, axis=0)
        np_y = numpy.exp(np_x) / numpy.sum(numpy.exp(np_x), axis=0)
        return np_y.T

    np_x = np_x - numpy.max(np_x) # オーバーフロー対策
    return numpy.exp(np_x) / numpy.sum(numpy.exp(np_x))


def mean_squared_error(np_y, t_label):
    u"""
    損失関数：2乗和誤差
    """
    return 0.5 * numpy.sum((np_y-t_label)**2)


def cross_entropy_error(np_y, t_label):
    u"""
    損失関数：交差エントロピー誤差
    """
    if np_y.ndim == 1:
        t_label = t_label.reshape(1, t_label.size)
        np_y = np_y.reshape(1, np_y.size)

    if t_label.size == np_y.size:
        t_label = t_label.argmax(axis=1)

    batch_size = np_y.shape[0]
    delta = 1e-7
    np_y = np_y + delta
    return -(numpy.sum(numpy.log(np_y[numpy.arange(batch_size), t_label])) / batch_size)
