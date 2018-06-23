# coding: UTF-8

import argparse
import time
import os
import numpy as np
from PIL import Image

import chainer

from networks.conditionalDCGAN import Generator

def main():
    parser = argparse.ArgumentParser(description='Generate One Image')
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--model_result_id', '-mri', default='none', type=str, help='ex. 180614_120722')
    parser.add_argument('--label', '-l', default='0;1.0;0;0;0;0;0;0;0;0', type=str)
    args = parser.parse_args()

    print('=== Generate Image ===')

    # 作業用path設定
    if args.model_result_id == 'none':
        raise ValueError('model_result_id is NOT defined')

    path = 'result/' + args.model_result_id + '/'
    os.chdir(path)

    ## モデルの各種設定
    with open('debug.txt', 'r') as f:
        text = f.readlines()

    for i in range(len(text)):
        string = text[i]

        if string.startswith('# n_hidden'):
            string = string.replace(' ', '')
            string = string.replace('\n', '')
            # 入力ノイズ次元数
            n_hidden = int(string.split(':')[1])
            print('n_hidden: {}'.format(n_hidden))

    # 学習済みGenerator設定
    gen = Generator(n_hidden=n_hidden)
    chainer.serializers.load_npz('generator.npz', gen)

    # cpu or gpu
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
    xp = chainer.cuda.cupy if args.gpu >= 0 else np

    # 入力ノイズデータ作成(バッチサイズ2)
    z = chainer.Variable(xp.asarray(gen.make_hidden(2)))

    # ラベル整形 str -> np.array
    a = args.label.split(';')
    label = xp.array(args.label.split(';'), dtype=xp.float32)

    label = xp.vstack((label, label)).reshape(2, 10, 1, 1)

    # 画像生成
    with chainer.using_config('train', False):
        x_fake = gen(z, label)

    x_fake = chainer.cuda.to_cpu(x_fake.data)
    x_fake = np.asarray(np.clip(x_fake * 255, 0.0, 255.0), dtype=np.uint8)

    _, _, H, W = x_fake.shape
    x = x_fake[0].reshape(H, W)

    preview_dir = 'images/test'
    preview_path = preview_dir + '/image_{}.png'.format(time.strftime('%y%m%d_%H%M%S', time.localtime()))
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)


if __name__ == '__main__':
    main()