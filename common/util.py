u"""汎用関数群"""
# coding: utf-8
import numpy


def shuffle_dataset(x_train, t_label):
    u"""データセットのシャッフルを行う

    引数:
    x_train 訓練データ
    t_label 教師データ

    戻り値:
    x_train t_label : シャッフルを行った訓練データと教師データ
    """
    permutation = numpy.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :] if x_train.ndim == 2 else x_train[permutation, :, :, :]
    t_label = t_label[permutation]

    return x_train, t_label

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    u"""画像から行列への変換

    引数:
    inumpyut_data -- (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h -- フィルターの高さ
    filter_w -- フィルターの幅
    stride -- ストライド
    pad -- パディング

    戻り値:
    col -- 2次元配列
    """
    num, channel, height, width = input_data.shape
    out_h = (height + 2*pad - filter_h)//stride + 1
    out_w = (width + 2*pad - filter_w)//stride + 1

    img = numpy.pad(
        input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = numpy.zeros((num, channel, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(num*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    u"""行列から画像への変換

    引数:
    col
    input_shape -- 入力データの形状（例：(10, 1, 28, 28)）
    filter_h -- フィルターの高さ
    filter_w -- フィルターの幅
    stride -- ストライド
    pad -- パディング

    戻り値:
    画像データ

    """
    num, channel, height, width = input_shape
    out_h = (height + 2*pad - filter_h)//stride + 1
    out_w = (width + 2*pad - filter_w)//stride + 1
    col = col.reshape(
        num, out_h, out_w, channel,
        filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = numpy.zeros(
        (num, channel, height + 2*pad + stride - 1, width + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:height + pad, pad:width + pad]
