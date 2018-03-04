# coding: utf-8
import sys, os
sys.path.append('common/')

import numpy as np
from collections import OrderedDict

from deconv import *
from leakyrelu import *
from tanh import *
from img_show import *


# filter_num <= 認識には使う。生成には使わない
def filter_num(input_length,
               pad_stride,
               pad,
               filter_length,
               filter_stride):

    filter_num = (input_length * pad_stride + pad * 2 ) - filter_length + 1 / filter_stride

    return int(filter_num)


# Generator
class Generator:
    """
    # ネットワーク構成
    deconv - relu - deconv - relu

    Parameters
    ----------
    input_size :
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNIST : 28*28=784）
    activation : 'relu'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    """

    def __init__(self,
                 input_dim=(100,1,1),
                 conv_param={'filter_size':5, # フィルター:5*5
                             'pad':2,         # パディング:0
                             'stride':1,      # ストライド:2
                             'pad_stride':2}, # 入力野間引き:2
                 output_size=784,  # 出力は28*28の確率分布
                 weight_init_std=0.01):


        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        filter_pad_stride = conv_param['pad_stride']
        input_size = 4
        output_height, output_width = 28, 28

        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

        # 重みの初期化
        self.params = {}

        # filter_num1 = filter_num(input_size, filter_pad_stride, filter_pad, filter_size, filter_stride)
        filter_num = 1
        W2_height, W2_width = output_height, output_width # 2層目の出力(ouput 28*28)
        W1_height, W1_width = W2_height/2, W2_width/2     # 1層目の出力(ouput 14*14)

        self.params['W1'] = weight_init_std * np.random.randn(filter_num, 1, filter_size, filter_size)
        self.params['b1'] = np.zeros(int(W1_height * W1_width))

        self.params['W2'] = weight_init_std * np.random.randn(filter_num, 1, filter_size, filter_size)
        self.params['b2'] = np.zeros(int(W2_height * W2_width))

        # レイヤの生成(2層ネットワーク)
        self.layers = OrderedDict()

        self.layers['Deconv1'] = Deconvolution(self.params['W1'], self.params['b1'],
                                             conv_param['stride'], conv_param['pad'], conv_param['pad_stride'])
        self.layers['Relu1'] = LeakyRelu()

        self.layers['Deconv2'] = Deconvolution(self.params['W2'], self.params['b2'],
                                             conv_param['stride'], conv_param['pad'], conv_param['pad_stride'])
        self.layers['Relu2'] = LeakyRelu()

        self.last_layer = Tanh()


    def gen(self, x):
        """
        画像の生成

        Param
        ----------
        x : 入力データ（乱数×10）

        Returns
        -------
        6画像を想定する確率分布
        """

        for layer in self.layers.values():
            x = layer.forward(x)

        x = self.last_layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxはdiscriminatorの出力データ、tは教師ラベル
        """
        if x.ndim == 1:
            t = t.reshape(1, t.size)
            x = x.reshape(1, x.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == x.size:
            t = t.argmax(axis=1)

        batch_size = x.shape[0]
        return -np.sum(np.log(x[np.arange(batch_size), t] + 1e-7)) / batch_size

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self):

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Deconv1'].dW, self.layers['Deconv1'].db
        grads['W2'], grads['b2'] = self.layers['Deconv2'].dW, self.layers['Deconv2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
