u"""機械学習用のデータセットを効率的に作成するためのモジュール"""
import os
import pickle
import numpy
from PIL import Image


def create_dataset_pkl(data_dir_path, image_size=100):
    u"""機械学習用のデータセットをpklファイルで出力する
    分類データを下記のようなディレクトリ構造で配置していることが前提
    .
    ├── 分類１──画像群
    ├── 分類２──画像群
    ├── 分類３──画像群
    └── 分類４──画像群

    .のパスを指定すること
    image_pkl -- numpy配列に変換した画像群
    label_pkl -- image_pklの正解ラベル

    引数:
    data_dir_path -- データセットが格納されているディレクトリ
    image_size -- 指定したサイズでデータセットをリサイズする

    戻り値:
    なし
    """
    #pklファイル出力用のリスト
    image_pkl = []
    label_pkl = []

    categories = os.listdir(data_dir_path)
    categories.sort()
    label = 0

    for category in categories:
        #分類フォルダまでのパスを作成
        category_dir_path = data_dir_path + category

        #分類フォルダ内の画像をnumpy配列に変換して取得する
        data_list = __get_category_data_list(
            category_dir_path, image_size, label)

        #画像とラベルをリストに格納
        image_pkl.extend(data_list)

        #ラベルを作成
        label_list = [label] * len(data_list)
        label_pkl.extend(label_list)

        label += 1

    #pklファイルで出力
    __output_pkl(image_pkl, label_pkl)


def restore_dataset(pkl_path="./dataset.pkl"):
    u"""
    create_dataset_pklで出力したpklファイルをプログラムで使用する形式に復元する

    引数:
    pkl_path -- create_dataset_pklで出力したpklファイルへのパス

    戻り値:
    x_train -- 訓練画像データのリスト
    t_label -- 教師ラベル
    """
    datas = []
    with open(pkl_path, 'rb') as file:
        datas = pickle.load(file)

    imageset = []
    labelset = []
    for data in datas:
        imageset.append(data[0])
        labelset.append(data[1])

    x_train = numpy.array(imageset)
    t_label = numpy.array(labelset)

    return x_train, t_label


def __get_category_data_list(category_dir_path, image_size, label):
    u"""
    分類ディレクトリの中にある画像ファイルをnumpy配列に変換してリストに入れる

    引数：
    category_dir_path -- 分類フォルダへのパス
    image_size -- 返還後の画像サイズ

    戻り値：
    分類ディレクトリ内にある画像ファイルをnumpy配列に変換したデータを格納リスト
    """

    #分類フォルダ内にある画像名を取得
    images = os.listdir(category_dir_path)

    #変換画像格納用のリスト
    data_list = []

    for image in images:
        #画像のファイルパスを生成
        image_path = category_dir_path + "/" + image

        #画像ファイルを開いて指定サイズにリサイズ、グレースケールに変換
        img = Image.open(image_path)
        img = img.resize((image_size, image_size)).convert('L')
        np_img = numpy.array(img)
        np_img = np_img.reshape(1, image_size, image_size)

        data_list.append((np_img, label))

    return data_list


def __output_pkl(image_pkl, label_pkl):
    """
    datasetとlabelをpklファイルで出力するサブルーチン
    """
    #numpy配列に変換してpklファイルとして保存
    dataset = numpy.array(image_pkl)
    label = numpy.array(label_pkl)

    dataset_output = open('dataset.pkl', 'wb')
    pickle.dump(dataset, dataset_output)
    dataset_output.close()

    label_output = open('label.pkl', 'wb')
    pickle.dump(label, label_output)
    label_output.close()
