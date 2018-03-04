# coding: utf-8
import math
import numpy as np

class LeakyRelu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        out = np.maximum(0.2 * x, x)
        return out

    def backward(self, dout):
        out = np.minimum(5*dout, dout)
        return out
