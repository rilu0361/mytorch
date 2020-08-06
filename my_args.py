"""
my_args.py
コマンドライン引数の管理
パラメータ管理
"""

import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='パラメータ指定')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--workers', default=0, type=int, help="CPUコア数") # 

    # data
    parser.add_argument('--data_path', default='./tmp_data', help="データのPATH")
    parser.add_argument('--checkpoint', default='checkpoint/',type=str, help="checkpointのPATH")

    # model
    parser.add_argument('--batch_size', default=300, type=int, help="batch size") # d:
    parser.add_argument('--numClasses', default=6, type=int) # d: 
    # parser.add_argument('--patience',   default=1, type=int)

    # image
    # parser.add_argument('--resize', default=224, type=int)
    # parser.add_argument('--preModel', default='resNet18',type=str) # resNet50
    # parser.add_argument('--imfeatDim', default=512, type=int) # d: 2048
    
    # text
    # parser.add_argument('--maxSeqlen', default=20, type=int)
    # parser.add_argument('--maxIngrs',  default=20, type=int)
    # parser.add_argument('--irnnDim', default=64, type=int) # d: 300 ingRNN hidden
    # parser.add_argument('--ingrW2VDim', default=100, type=int) # d:300 ingRNN input_size
    # parser.add_argument('--srnnDim', default=256, type=int) # d:1024 stRNN hidden_size
    # parser.add_argument('--stDim',   default=100, type=int) # d:1024 stRNN input_size 現状word2vecのサイズにならざるを得ないのは実装が異なるため。ただ加算して手順ベクトルを作成.

    # im2recipe model
    # parser.add_argument('--semantic_reg', default=True,type=bool)
    # parser.add_argument('--embDim',     default=256, type=int) # d: 1024 # 共通の埋め込みcos前

    # parser.add_argument('--nRNNs', default=1, type=int)
    # parser.add_argument('--maxImgs', default=5, type=int)

    # training 
    parser.add_argument('--epochs', default=1000, type=int) # d: 720
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--resume', default='./checkpoint/epoch005_val211.000.pth.tar', type=str)
    # 損失関数
    parser.add_argument('--cos_weight', default=0.98, type=float) # d: 0.98
    parser.add_argument('--cls_weight', default=0.02, type=float) # d: 0.01
    # 最適化関数
    parser.add_argument('--lr', default=0.001, type=float) # d: 0.0001
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    # parser.add_argument('--ingrW2V', default='data/vocab.bin',type=str)
    # parser.add_argument('--valfreq', default=10,type=int)  

    # 切り替え?
    # parser.add_argument('--freeVision', default=False, type=bool)
    # parser.add_argument('--freeRecipe', default=True,  type=bool)
    
    # test
    # parser.add_argument('--path_results', default='results/', type=str)
    # parser.add_argument('--model_path', default='snapshots/model_e220_v-4.700.pth.tar', type=str)
    # parser.add_argument('--test_image_path', default='chicken.jpg', type=str)    

    # MedR / Recall@1 / Recall@5 / Recall@10
    # parser.add_argument('--embtype', default='image', type=str) # [image|recipe] query type
    # parser.add_argument('--medr', default=100, type=int) # d: 1000 

    # dataset
    # parser.add_argument('--maxlen', default=20, type=int)
    # parser.add_argument('--vocab', default = 'vocab.txt', type=str)
    # parser.add_argument('--dataset', default = '../data/recipe1M/', type=str)
    # parser.add_argument('--sthdir', default = '../data/', type=str)

    return parser




