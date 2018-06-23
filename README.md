# conditionalDCGAN
MNIST��conditionalDCGAN�Ő�������v���O����

## �g����
* �w�K

```python ctreate_dataset.py```

mnist��pickle�ɕۑ����Ďg�p���Ă��܂�.

```python train.py```

Trainer���g��Ȃ��L�q���@�ł�.

* �摜����

```python test.py -mri result_id -l class_label```

�w�K�ς݃��f������摜��1���������܂�. 

    * result_id�ɂ͗��p������generator.py���ۑ����ꂽresult�����̃t�H���_�����w��

    * class_label�ɂ͐����������N���X�̃��x��(0�`9)���w��

        * (ex) �N���X0 --> 1;0;0;0;0;0;0;0;0;0

```test_sample.bat```�����s�����, �N���X0�`9�w��摜��,
�S��0�w��, �S��1�w��, 1��3��0.5���w�肵���摜�𐶐��ł��܂�.

## �����
* python==3.6.1
* chainer==3.5.0
* cupy==2.5.0
* cuda==v8.0
* cuDNN==v6
* GPU: GeForce GTX 1080

## �T���v��
![sample](./mnist_sample.gif)

## �Q�l
���L�̃T�C�g���Q�l�ɒv���܂���.

https://qiita.com/lyakaap/items/a9ae5d91464e72774093
