u"""機械学習用汎用レイヤー"""
# coding: utf-8
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.util import im2col, col2im


class Relu:
    u"""Relu
    インスタンス変数:
    mask -- True/Falseからなるnumpy配列 0以下をTrue,それ以外をFalseとして保持する
    """
    def __init__(self):
        self.mask = None

    def forward(self, x_train):
        u"""順方向伝播

        x_train -- numpy配列
        """
        self.mask = (x_train <= 0)
        out = x_train.copy()
        out[self.mask] = 0

        return out

    def backward(self, d_out):
        u"""逆方向伝播

        differential_out -- numpy配列
        """
        d_out[self.mask] = 0
        d_x = d_out

        return d_x


class Affine:
    u"""Affine
    インスタンス変数:
    weight -- 重み
    bias -- バイアス
    """
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

        self.x_train = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.d_weight = None
        self.d_bias = None

    def forward(self, x_train):
        u"""順伝播
        """
        # テンソル対応
        self.original_x_shape = x_train.shape
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.x_train = x_train

        out = np.dot(self.x_train, self.weight) + self.bias

        return out

    def backward(self, d_out):
        u"""逆伝播
        """
        d_x = np.dot(d_out, self.weight.T)
        self.d_weight = np.dot(self.x_train.T, d_out)
        self.d_bias = np.sum(d_out, axis=0)

        d_x = d_x.reshape(*self.original_x_shape)
        return d_x


class SoftmaxWithLoss:
    u"""ソフトマックスレイヤ
    """
    def __init__(self):
        self.loss = None
        self.y_output = None # softmaxの出力
        self.t_label = None # 教師データ

    def forward(self, x_train, t_label):
        u"""順伝播
        """
        self.t_label = t_label
        self.y_output = softmax(x_train)
        self.loss = cross_entropy_error(self.y_output, self.t_label)

        return self.loss

    def backward(self, dout=1):
        u"""逆伝播
        """
        batch_size = self.t_label.shape[0]
        if self.t_label.size == self.y_output.size:
            d_x = (self.y_output - self.t_label) / batch_size
        else:
            d_x = self.y_output.copy()
            d_x[np.arange(batch_size), self.t_label] -= 1
            d_x = d_x / batch_size

        return d_x

class Convolution:
    u"""Convレイヤ
    """
    def __init__(self, weight, bias, stride=1, pad=0):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x_train = None
        self.col = None
        self.col_w = None

        # 重み・バイアスパラメータの勾配
        self.d_weight = None
        self.d_bias = None

    def forward(self, x_train):
        u"""順伝播
        """
        f_num, channel, f_height, f_weight = self.weight.shape
        num, channel, height, weight = x_train.shape

        out_h = 1 + int((height + 2*self.pad - f_height) / self.stride)
        out_w = 1 + int((weight + 2*self.pad - f_weight) / self.stride)

        col = im2col(x_train, f_height, f_weight, self.stride, self.pad)
        col_w = self.weight.reshape(f_num, -1).T

        out = np.dot(col, col_w) + self.bias
        out = out.reshape(num, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x_train = x_train
        self.col = col
        self.col_w = col_w

        return out

    def backward(self, d_out):
        u"""逆伝播
        """
        f_num, channel, f_height, f_width = self.weight.shape
        d_out = d_out.transpose(0, 2, 3, 1).reshape(-1, f_num)

        self.d_bias = np.sum(d_out, axis=0)
        self.d_weight = np.dot(self.col.T, d_out)
        self.d_weight = self.d_weight.transpose(1, 0).reshape(
            f_num, channel, f_height, f_width)

        d_col = np.dot(d_out, self.col_w.T)
        d_x = col2im(
            d_col, self.x_train.shape, f_height, f_width,
            self.stride, self.pad)

        return d_x


class Pooling:
    u"""プーリング層
    """
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x_train = None
        self.arg_max = None

    def forward(self, x_train):
        u"""順伝播
        """
        num, channel, height, width = x_train.shape
        out_h = int(1 + (height - self.pool_h) / self.stride)
        out_w = int(1 + (width - self.pool_w) / self.stride)

        col = im2col(x_train, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(num, out_h, out_w, channel).transpose(0, 3, 1, 2)

        self.x_train = x_train
        self.arg_max = arg_max

        return out

    def backward(self, d_out):
        u"""逆伝播
        """
        d_out = d_out.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        d_max = np.zeros((d_out.size, pool_size))
        d_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_out.flatten()
        d_max = d_max.reshape(d_out.shape + (pool_size,))

        d_col = d_max.reshape(
            d_max.shape[0] * d_max.shape[1] * d_max.shape[2], -1)
        d_x = col2im(
            d_col, self.x_train.shape, self.pool_h,
            self.pool_w, self.stride, self.pad)

        return d_x
