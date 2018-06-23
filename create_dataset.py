# coding: UTF-8

import numpy as np
import pickle
import chainer


def main():
    train, _ = chainer.datasets.get_mnist(withlabel=True, ndim=1, scale=255.)

    print('=== create MNIST dataset ===')

    datas = None
    labels = None
    for i in range(len(train)):
        data = train[i][0].reshape(1,784)
        label = train[i][1]
        if datas is None:
            datas = np.vstack(([data]))
            labels = np.hstack(([label]))
        else:
            datas = np.vstack((datas, data))
            labels = np.hstack((labels, label))

    print('datas.shape: ', datas.shape)
    print('labels.shape: ', labels.shape)

    dataset = {}
    dataset['datas'] = datas
    dataset['labels'] = labels

    with open('./mnist.pkl', 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()