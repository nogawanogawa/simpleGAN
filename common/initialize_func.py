# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
from dataset.mnist import load_mnist

# mnistのデータを取得する
def get_data():

    # ２次元データのままデータを取得
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)

    return (x_train, t_train), (x_test, t_test)
