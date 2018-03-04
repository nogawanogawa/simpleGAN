# coding: utf-8
import numpy as np
from activation_func import *
from img_show import *

# 二次元データ　<=> 一次元データ変換用クラス
from common.util import im2col, col2im

# 逆畳み込み層
class Deconvolution:
    def __init__(self,
                 W, # ネットワーク
                 b, # バイアス
                 stride=1, #ストライド
                 pad=2,# パディングのサイズ
                 pad_stride=2): # 入力データの間引き
        # ネットワークの形状
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.pad_stride = pad_stride

        #　初期化（backward時に使用）
        self.x = None     # 入力(2D)
        self.col = None   # 入力(1D)
        self.col_W = None # 入力(1Dの転置)

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None


    def convolution(self,x): # 入力信号

        # フィルターと出力の形状のセットアップ
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # フィルターのサイズに分割、1次元のベクトルに整形
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # ベクトルの積和演算
        out = np.dot(col, col_W)
        out = out.T + self.b

        # 出力の転置（並べ替え）
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        #　元の形状を記憶
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def forward(self,x): # 入力信号

        # フィルターと出力の形状
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # transpose処理
        x_padded = np.zeros((1, 1, H*2, W*2), dtype=np.float32)
        x_padded[:, :, ::self.pad_stride, ::self.pad_stride] = x

        out = self.convolution(x_padded)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = dout.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        # affine層と同様の逆伝播
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        # transpose処理（縮小方向）
        H_ = int(H/2)
        W_ = int(W/2)
        dx_ = np.zeros((1, 1, H_, W_), dtype=np.float32)
        dx_ = dx[:, :, ::self.pad_stride, ::self.pad_stride]

        return dx_
