# coding: UTF-8

import os
import numpy as np
from PIL import Image

import chainer

def out_generated_image(gen, rows, cols, seed, iteration, xp):
    np.random.seed(seed)
    n_images = rows * cols

    z = chainer.Variable(xp.asarray(gen.make_hidden(n_images)))
    # seedのクリア
    np.random.seed()

    # ラベル0*10 -> 1*10 -> ...なるarrayを用意
    labels = [i for i in range(rows) for j in range(cols)]

    # one_hotラベルをバッチ数分作成
    from train import create_one_hot_label
    labels = xp.asarray(create_one_hot_label(10, labels)).reshape(n_images, 10, 1, 1)

    with chainer.using_config('train', False):
        x = gen(z, labels)

    x = chainer.cuda.to_cpu(x.data)

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)

    _, _, H, W = x.shape

    x = x.reshape((rows, cols, 1, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W))

    preview_dir = 'images/train'
    preview_path = preview_dir + '/image_iteration_{:0>8}.png'.format(iteration)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)