u"""７層ニューラルネットワークのプログラム"""
# coding: utf-8
import pickle
from collections import OrderedDict
import numpy
from common.layers import Relu, Pooling, Convolution, Affine, SoftmaxWithLoss
from common.gradient import numerical_gradient


class SimpleConvNet:
    u"""単純なConvNet

    conv - relu - pool - affine - relu - affine - softmax

    引数：
    input_size -- 入力画像のサイズ
    hidden_size_list -- 隠れ層のニューロンの数のリスト
    output_size -- 分類種別の個数
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={
                     'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / \
                            filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * \
                            (conv_output_size/2))

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            numpy.random.randn(
                                filter_num, input_dim[0],
                                filter_size, filter_size)
        self.params['b1'] = numpy.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            numpy.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = numpy.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            numpy.random.randn(hidden_size, output_size)
        self.params['b3'] = numpy.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(
            self.params['W1'], self.params['b1'],
            conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

        #self.load_params(file_name="params_0309.pkl")

    def predict(self, x_train):
        """
        引数：
        x_train -- 入力画像
        """
        for layer in self.layers.values():
            x_train = layer.forward(x_train)

        return x_train

    def loss(self, x_train, t_label):
        """損失関数を求める
        x_train -- 入力画像
        t_label -- 教師ラベル
        """
        predicted_train = self.predict(x_train)
        return self.last_layer.forward(predicted_train, t_label)

    def accuracy(self, x_train, t_label, batch_size=100):
        u"""認識精度を求める
        x_train -- 入力画像
        t_label -- 教師ラベル
        batch_size -- バッチサイズ
        """
        if t_label.ndim != 1:
            t_label = numpy.argmax(t_label, axis=1)

        acc = 0.0

        for i in range(int(x_train.shape[0] / batch_size)):
            print("accuracy:" + str(i) + "回目")
            t_x = x_train[i*batch_size:(i+1)*batch_size]
            t_t = t_label[i*batch_size:(i+1)*batch_size]
            predicted = self.predict(t_x)
            predicted = numpy.argmax(predicted, axis=1)
            acc += numpy.sum(predicted == t_t)

        return acc / x_train.shape[0]

    def numerical_gradient(self, x_train, t_label):
        """勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_w = lambda w: self.loss(x_train, t_label)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(
                loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(
                loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x_train, t_label):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x_train, t_label)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].d_weight, self.layers['Conv1'].d_bias
        grads['W2'], grads['b2'] = self.layers['Affine1'].d_weight, self.layers['Affine1'].d_bias
        grads['W3'], grads['b3'] = self.layers['Affine2'].d_weight, self.layers['Affine2'].d_bias

        return grads

    def save_params(self, file_name="params.pkl"):
        u"""学習したパラメータを保存する
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as file:
            pickle.dump(params, file)

    def load_params(self, file_name="params.pkl"):
        u"""学習済みパラメータをロードする
        """
        with open(file_name, 'rb') as file:
            params = pickle.load(file)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
