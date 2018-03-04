# coding: utf-8
import numpy as np

#tanh関数
class Tanh:

    def __init__(self):
        self.y = None

    def forward(self, x):
        self.x = x
        y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        y = y * 255
        #y = (y+1)/2 # 0 ~ 1に圧縮
        return y

    def backward(self, dout=1):
        dx = 1 / (np.cosh(self.x) * np.cosh(self.x) * 255)
        return dx
