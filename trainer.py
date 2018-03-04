 # coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pdb as p
from optimizer import *
from img_show import *


#ニューラルネットの訓練を行うクラス
class Trainer:

    def __init__(self,
                 generator,
                 discriminator,
                 x_train, t_train, x_test, t_test, # トレーニング/テストデータ
                 epochs,
                 mini_batch_size=100,
                 optimizer_gen='SGD',
                 optimizer_disc='SGD',
                 optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None,
                 verbose=True):

        self.generator = generator
        self.discriminator = discriminator
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        self.batch_mask = None

        # optimzerの候補
        optimizer_class_dict = {'sgd':SGD,
                                'momentum':Momentum,
                                'nesterov':Nesterov,
                                'adagrad':AdaGrad,
                                'rmsprpo':RMSprop,
                                'adam':Adam}

        # 最適化手法の設定
        self.optimizer_gen = optimizer_class_dict[optimizer_gen.lower()](**optimizer_param)
        self.optimizer_disc = optimizer_class_dict[optimizer_disc.lower()](**optimizer_param)

        # その他セットアップ
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        self.input_size = 7

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):

        #rand = np.random.rand(self.input_size*self.input_size)
        #rand = np.random.uniform(-0.25, 0.25, size=self.input_size*self.input_size)
        rand = np.random.uniform(0.0, 0.5, size=self.input_size*self.input_size)
        input_gen = np.reshape(rand,(1,1,self.input_size,self.input_size))

        # 1イテレーションの実行
        gen_result = self.generator.gen(input_gen) # 画像生成

        # 教師データを混ぜ込み
        x_batch = self.x_train[self.batch_mask] # 教師データの取得

        disc_src = np.vstack((x_batch, gen_result))
        t = np.zeros((2,2))
        t_ = np.zeros((2,2))


        # 先頭が教師データ、それ以降が生成した偽データ
        t[0, 0]  = 1 # 先頭が正解(1)
        t[1:, 1] = 1 # その後ろが不正解(1)

        t_[1:, 0]  = 1 # 先頭が正解(1)
        t_[0, 1] = 1 # その後ろが不正解(1)

        predict = self.discriminator.predict(disc_src) # 画像推定

        predict_ , disc_loss = self.discriminator.last_layer.forward(predict, t)
        gen_loss = self.generator.loss(predict_, t_)

        disc_dout = self.discriminator.gradient()
        gen_dout  = self.generator.gradient()

        self.optimizer_gen.update(self.generator.params, gen_dout) # generatorの学習
        self.optimizer_disc.update(self.discriminator.params, disc_dout) # discriminatorの学習　


        # ロス（正解との差分）を計算・記録
        # loss = self.network.loss(x_batch, t_batch)
        # self.train_loss_list.append(loss)
        # if self.verbose: print("train loss:" + str(loss))

        # 学習データをちょうど一巡した場合にはグラフ出力用にテストを実施
        """
        if self.current_iter % self.iter_per_epoch == 0:

            # エポックをインクリメント
            self.current_epoch += 1

            #　教師・テストデータを取得
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            # 予測精度の算出
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        """
        if self.current_iter % 100 == 0:
            img_show(gen_result.reshape(28,28))
            img_show(x_batch.reshape(28,28))

        self.current_iter += 1

    def train(self):
        # 学習に使用するデータをランダムに選択
        self.batch_mask = np.random.choice(self.train_size, 1)

        for i in range(self.max_iter):
            print("iteration : " + str(i) )
            self.train_step()

        # generatorの精度確認
        rand = np.random.uniform(-0.25, 0.25, size=self.input_size*self.input_size)
        input_gen = np.reshape(rand,(1,1,self.input_size,self.input_size))
        gen_result = self.generator.gen(input_gen) # 画像生成
        img_show(gen_result.reshape(28,28))
        print(gen_result)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
