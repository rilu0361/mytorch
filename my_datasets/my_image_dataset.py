from PIL import Image, ImageFile
import torch.utils.data as data
import os
from pathlib import Path

class ImageDataset(data.Dataset):
    """
    画像のデータセットの作成

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, path_list, transform=None, phase='train'):
        self.path_list = path_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.path_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''
        img_path = self.path_list[index] # index番目の画像をロード
        img = Image.open(img_path) # [高さ][幅][色RGB]
        # 画像の前処理を実施
        img_transformed = self.transform(img) # torch.Size([frame, 3, 224, 224])
        # print(img_transformed.size())
        # ラベルの取得
        label = self.get_label(img_path) # フォルダ名からラベルを取得する
        return img_transformed, label

    def get_label(self, video_path):
        label = int(video_path.split("_")[-2]) - 1
        return label

