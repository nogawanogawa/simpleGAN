# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from PIL import Image

# 28 * 28のサイズで出力
def img_show(img):
    #img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
