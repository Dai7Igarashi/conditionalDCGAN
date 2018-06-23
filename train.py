# coding:UTF-8

import argparse
import os
import numpy as np
import time
import logging
import pickle

import chainer
import chainer.functions as F
from chainer.computational_graph import build_computational_graph

from networks.conditionalDCGAN import Generator, Discriminator
from visualize import out_generated_image


class Iterator(object):
    def __init__(self, dataset, batchsize):
        self.dataset = dataset
        self.batchsize = batchsize

    @staticmethod
    def get_index(labels, shuffle):
        index = np.arange(len(labels))
        if shuffle:
            np.random.seed(int(time.clock()*1000))
            np.random.shuffle(index)
        return index

    @staticmethod
    def get_mini_batch(dataset, ref):
        return {'datas': dataset['datas'][ref], 'labels': dataset['labels'][ref]}

    def __call__(self, shuffle=True):
        self.index = self.get_index(self.dataset['labels'], shuffle)
        self.indexed_labels = self.dataset['labels'][self.index]

        for i in range(0, len(self.dataset['labels']), self.batchsize):
            ref = self.index[i:i+self.batchsize]
            yield self.get_mini_batch(self.dataset, ref)



def set_logger(logger_name, save_path):
    logger = logging.getLogger(logger_name + save_path)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('{}.txt'.format(logger_name), 'w')
    stream_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# one-hotラベルの生成
def create_one_hot_label(class_num, label):
    return np.eye(class_num)[label]


def main():
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('--batchsize', '-b', default=50, type=int)
    parser.add_argument('--epoch', '-e', default=1000, type=int)
    parser.add_argument('--gpu', '-g', default=0, type=int)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--n_hidden', '-n', default=100, type=int)
    parser.add_argument('--snapshot_interval', default=1000, type=int)
    args = parser.parse_args()

    print('=== DCGAN ===')

    # rootディレクトリ退避
    root = os.getcwd()

    # 出力フォルダ作成
    if not os.path.exists(args.out):
        print('** create result')
        os.mkdir(args.out)

    save_path = args.out + '/' + time.strftime('%y%m%d_%H%M%S', time.localtime())
    os.makedirs(save_path)

    os.chdir(save_path)

    # loggerの設定
    logger_names = ['losses', 'debug']
    loggers = {}
    for logger_name in logger_names:
        loggers[logger_name] = set_logger(logger_name, save_path)

    loggers['debug'].debug('# batchsize: {}'.format(args.batchsize))
    loggers['debug'].debug('# epoch: {}'.format(args.epoch))
    loggers['debug'].debug('# n_hidden: {}'.format(args.n_hidden))
    loggers['debug'].debug('')

    # 学習用のモデル設定
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()

    # cpu or gpu
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()
    xp = chainer.cuda.cupy if args.gpu >= 0 else np

    def make_optimizer(model, alpha=0.002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # MNISTデータ読み込み
    # contents['datas'] -> (60000, 784)
    # contents['labels] -> (60000,)
    with open(root + '/mnist.pkl', 'rb') as f:
        contents = pickle.load(f)

    loggers['debug'].debug('# data_size: {}'.format(contents['datas'].shape[0]))
    loggers['debug'].debug('')

    # バッチ作成
    iterator_train = Iterator(contents, args.batchsize)

    # 学習ループ
    iteration = 1
    for epoch in range(1, args.epoch+1):
        start_time = time.time()
        print('# epoch   gen/loss   dis/loss')

        # ミニバッチ学習
        for yielded in iterator_train(shuffle=True):
            batch = yielded
            variables = {}

            # データ整形 --- 784 -> 1ch, 28*28
            b_size = batch['datas'].shape[0]
            d = batch['datas'].reshape(b_size, 1, 28, 28)
            datas = chainer.Variable(xp.asarray(d, dtype=xp.float32))
            variables['datas'] = datas
            # 画像と対応したラベルを入れて学習させる
            labels = batch['labels']

            # one_hotラベルをバッチ数分作成
            labels = xp.asarray(create_one_hot_label(10, labels)).reshape(b_size, 10, 1, 1)

            # 勾配クリア
            gen.cleargrads()
            dis.cleargrads()

            ## 順伝播(DCGAN)
            # x_real = variables['datas'] / 255.
            x_real = variables['datas'] / 255.
            # Discriminatorの出力値(本物入力)
            y_real = dis(x=x_real, label=labels)
            # 入力ノイズデータ作成
            z = chainer.Variable(xp.asarray(gen.make_hidden(b_size)))
            # Generatorの出力値(ノイズ入力)
            x_fake = gen(z=z, label=labels)
            # Discriminatorの出力値(偽物入力)
            y_fake = dis(x=x_fake, label=labels)

            ## Discriminatorの誤差関数
            # 本物画像に対して本物(1)を出力させたい
            # 本物を本物と判定するほどL1は小さくなる
            L1 = F.sum(F.softplus(-y_real)) / b_size
            # 偽物画像に対して偽物(0)を出力させたい
            # 偽物を偽物と判定するほどL2は小さくなる
            L2 = F.sum(F.softplus(y_fake)) / b_size
            dis_loss = L1 + L2

            ## Generatorの誤差関数
            # 偽物画像を入力した時のDiscriminatorの出力を本物(1)に近づける
            # 偽物で本物と判定するほどlossは小さくなる
            gen_loss = F.sum(F.softplus(-y_fake)) / b_size

            loggers['losses'].debug('# epoch: {}, iteration{}'.format(epoch, iteration))
            loggers['losses'].debug('gen/loss: {} dis/loss: {}'.format(gen_loss.data, dis_loss.data))

            print('{}   {}   {}'.format(epoch, gen_loss.data, dis_loss.data))

            # 誤差逆伝播 -> 重み更新
            dis_loss.backward()
            opt_dis.update()

            gen_loss.backward()
            opt_gen.update()

            if iteration % args.snapshot_interval == 0:
                out_generated_image(gen, 10, 10, 0, iteration, xp)
            iteration += 1

        passed_time = time.time() - start_time
        print('*** passed time in this epoch: {}[sec]'.format(passed_time))

        loggers['debug'].debug('# epoch: {}'.format(epoch))
        loggers['debug'].debug('# passed_time: {}[sec]'.format(passed_time))

    print('=== Save Model ====')
    gen.to_cpu()
    chainer.serializers.save_npz('./generator.npz', gen)
    dis.to_cpu()
    chainer.serializers.save_npz('./discriminator.npz', dis)

    # computational_graph作成
    print('=== Draw Computational Graph ===')
    _val_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
    _fanc_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}
    with open('computational_graph.dot', 'w') as o:
        g = build_computational_graph([gen_loss, dis_loss], variable_style=_val_style,
                                      function_style=_fanc_style)
        o.write(g.dump())

if __name__ == '__main__':
    main()