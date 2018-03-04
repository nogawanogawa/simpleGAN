# coding: utf-8
import sys, os
sys.path.append('common/')

import numpy as np
import matplotlib.pylab as plt

from generator import Generator
from discriminator import Discriminator
from trainer import Trainer
from initialize_func import get_data


# mnistデータの取得
(x_train, t_train), (x_test, t_test) = get_data()

#epochの数を設定
max_epochs = 3

# networkの初期化(CNN)
generator = Generator(input_dim=(100), # 乱数
                      conv_param = {'filter_size': 5, 'pad': 2, 'stride': 1, 'pad_stride':2},
                      output_size=(1,28,28), # 出力（グレースケール、縦、横）
                      weight_init_std=0.01) # ネットワークの重みの初期値

discriminator = Discriminator(input_dim=(1,28,28), # 画像のサイズ（グレースケール、縦、横）
                      conv_param = {'filter_num':1 ,'filter_size': 5, 'pad': 0, 'stride': 1},
                      output_size=2, #出力（教師データか生成データの二択の確率分布）
                      weight_init_std=0.01) # ネットワークの重みの初期値


trainer = Trainer(generator,
                  discriminator,
                  x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer_gen='Adam',
                  optimizer_disc='Adam',
                  optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)

trainer.train()

"""
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""
