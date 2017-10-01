"""サンプルプログラム"""
# coding: utf-8
import numpy
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from data_set_util import restore_dataset


def sample():
    """適当に学習させる
    create_dataset_pklを使って訓練データとテストデータを作ってそれぞれ
    train.pklとtest.pklにリネームすれば画像認識の学習を行う
    SimpleConvNetのパラメータoutput_sizeは分類の数を設定する必要がある
    """

    x_train, t_train = restore_dataset("./train.pkl")
    x_test, t_test = restore_dataset("./test.pkl")

    max_epochs = 20

    network = SimpleConvNet(
        input_dim=(1, 100, 100),
        conv_param={'filter_num': 30, 'filter_size': 15, 'pad': 0, 'stride': 1},
        hidden_size=300, output_size=15, weight_init_std=0.01)

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # パラメータの保存
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = numpy.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    sample()
