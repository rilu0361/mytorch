# torch テンプレート

とても工事中です.

## 概要
pytorchで深層学習を実行するためのプログラム.

## 環境構築

`my_env.yml` を参照.

Anaconda環境下であれば以下のコマンドで`tmptorch`という名前の仮想環境が構築されます.

```sh
conda env create -f=my_env.yml
```

仮想環境を削除したい場合は以下.
```sh
conda remove -n tmptorch --a
```

pytorchはCUDAなどのバージョンを見ながら,下記公式サイトに従ってinstallしてください.

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)


## 実行方法



## フォルダ構成

```
.
├── README.md
├── 00_train.py
├── 10_test.py
├── my_args.py
├── my_util.py
├── my_text_dataset.py
├── my_image_dataset.py
├── my_audio_dataset.py
├── my_text_transform.py
├── my_image_transform.py
├── my_audio_transform.py
├── my_model.py
├── my_cnn.py
├── my_rnn.py
├── my_fc.py
├── tmp_data
│      ├──image
│      │    ├──train
│      │    ├──valid
│      │    └──test
│      ├──text
│      └──audio
├── util
├── my_env.yml
└── LICENSE

```

`tmp_data` サンプルのデータを配置しておく.
動作確認などに使用する.



