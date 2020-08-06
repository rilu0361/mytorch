"""
00_train.py
学習用プログラム。
"""
# 一般
import os
import time
import random
import shutil, os
import numpy as np
from tqdm import tqdm
# pytorch関連
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter  # tensorboard
# 自作プログラムの読込
from my_args       import get_parser
# from my_transform  import ImageTransform 
# from my_dataset    import RecipeDataset
# from my_model      import Im2RecipeNet

# コマンドライン===============================================================
parser = get_parser()
opts   = parser.parse_args() # opts.xxx でxxxのパラメータの呼び出し
# =============================================================================

'''
初期条件確認
'''
print("LOG : 開始現在 ", time.strftime("%Y/%m/%d %H:%M", time.strptime(time.ctime())))
start_time = time.time()
print("- - - - - - - - - -")
# tensorboard準備
if os.path.exists(opts.tensorboard): # TODO:パラメータ設定on/off
    shutil.rmtree(opts.tensorboard)
    print("FILE : ",opts.tensorboard,"を削除しました.")
writer = SummaryWriter(log_dir=opts.tensorboard) # tbxのインスタンス生成.フォルダ自動生成
print("FILE : ",opts.tensorboard,"を作成しました.")

# モデルの保存用フォルダの作成
if not os.path.exists(opts.checkpoint):
    os.mkdir(opts.checkpoint)
    print("FILE : ", opts.checkpoint, "を作成しました.")
else :
    shutil.rmtree(opts.checkpoint)
    os.mkdir(opts.checkpoint)
    print("FILE : ", opts.checkpoint, "を削除し再作成しました.")
print("INFO : checkpoint path -> ", opts.checkpoint)

# GPU確認
if not(torch.cuda.device_count()):
    DEVICE = torch.device(*('cpu',0))
else:
    torch.cuda.manual_seed(opts.seed)
    DEVICE = torch.device(*('cuda',0))
print("INFO : 使用するデバイス -> ", DEVICE)
# DEVICE = "cpu"

# cudnnの自動チューナー:場合によって速くなったり遅くなったり(入力サイズが常に一定だといいらしいが)
cudnn.benchmark = True 
print("LOG : 初期確認終了 経過時間:{:.2f}".format(time.time()-start_time))
print("- - - - - - - - - -")
'''
モデル定義
'''
print("LOG : モデル定義...")
# モデル定義
print(DEVICE)
model = Im2RecipeNet().to(DEVICE, non_blocking=True)
print(model)
# Loss関数定義
cosine_crit = nn.CosineEmbeddingLoss(0.1).to(DEVICE)
# カテゴリ分類ありの場合
if SEMANTIC_REG: 
    weights_class = torch.Tensor(NUM_CLASSES).fill_(1)
    weights_class[0] = 0 # the background class is set to 0
    class_crit = nn.CrossEntropyLoss(weight=weights_class).to(DEVICE)
    criterion = [cosine_crit, class_crit]
else:
    criterion = cosine_crit

# optimizer定義
# # creating different parameter groups
vision_params = list(map(id, model.visionMLP.parameters()))
base_params   = filter(lambda p: id(p) not in vision_params, model.parameters())
   
# optimizer = torch.optim.Adam(model.parameters(), lr=LARNING_RATE, eps=1e-08, weight_decay=WEIGHT_DECAY)
optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.visionMLP.parameters(), 'lr': LARNING_RATE*FREEVISION }
            ], lr=LARNING_RATE*FREERECIPE)

# 保存したものがあれば呼び出す
print("LOG : checkpointの呼び出し...")
if RESUME:
    if os.path.isfile(RESUME):
        print("=> loading checkpoint '{}'".format(RESUME))
        checkpoint = torch.load(RESUME)
        START_EPOCH = checkpoint['epoch']
        best_val = checkpoint['best_val']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(RESUME, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(RESUME))
        best_val = float('inf') 
else:
    best_val = float('inf') 
    print("=> no checkpoint found")

# モデルは検証データで一番良かったときのみ保存する
valtrack = 0

'''
データローダー定義
'''
print("LOG : データローダー定義...")
# 学習データ
print("INFO : train_data = " , os.path.join(DATA_PATH, "train/menu.csv"))
trainset    = RecipeDataset(os.path.join(DATA_PATH, "train_m"),
                            os.path.join(IMG_PATH, "train_image"), 
                            transform=image_transform, phase="train",
                            sem_reg=opts.semantic_reg)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=WORKERS, pin_memory=True)
print('LOG : Training loader prepared.')

# # 学習データ
# print("INFO : train_data = " , os.path.join(DATA_PATH, "valid/menu.csv"))
# trainset    = RecipeDataset(os.path.join(DATA_PATH, "valid"),
#                             os.path.join(IMG_PATH, "valid_image"), 
#                             transform=image_transform, phase="train",
#                             sem_reg=opts.semantic_reg)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
#                             shuffle=True, num_workers=WORKERS, pin_memory=True)
# print('LOG : Training loader prepared.')

# 検証用データ
validset    = RecipeDataset(os.path.join(DATA_PATH, "valid_m"),
                            os.path.join(IMG_PATH, "valid_image"),
                            transform=image_transform, phase="valid",
                            sem_reg=opts.semantic_reg)
validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=WORKERS, pin_memory=True)
print('LOG : Validation loader prepared.')

'''
tensorboard設定
'''
# tensorboardへの記録用関数
def write_tbx(epoch, cos_loss, img_loss, rec_loss, val_medr, val_recall):
    print(epoch, cos_loss, img_loss, rec_loss, val_medr)
    writer.add_scalar('train/cos_loss', cos_loss, epoch) 
    writer.add_scalar('train/img_loss', img_loss, epoch)
    writer.add_scalar('train/rec_loss', rec_loss, epoch)
    writer.add_scalar('valid/medr',     val_medr, epoch)
    writer.add_scalar('valid/recall@1',  val_recall[1],  epoch)
    writer.add_scalar('valid/recall@5',  val_recall[5],  epoch)
    writer.add_scalar('valid/recall@10', val_recall[10], epoch)

'''
学習ループ定義
'''
# 学習用関数(1epoch)
def one_train(loader, model, criterion, optimizer, epoch):
    print("LOG : training phase , epoch = ", epoch)
    # 各値初期化
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()
    data_num  = len(loader.dataset)              # テストデータの総数
    pbar = tqdm(total=int(data_num/BATCH_SIZE))  # プログレスバー設定
    # 学習開始
    model.train()          # モデルを学習モードに設定
    for batch, (inputs, targets) in enumerate(loader): 
        # データをdeviceに載せる (image, inst, len(inst), ingr, len(ingr)), [target, img_class, rec_class]
        input_var  = [data.to(DEVICE, non_blocking=True) for data in inputs]
        target_var = [data.to(DEVICE, non_blocking=True) for data in targets]
        outputs = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4]) # モデルから出力を得る 
        
        # Lossの計算 カテゴリ分類のあるなしで場合分け
        if SEMANTIC_REG:
            cos_loss = criterion[0](outputs[0], outputs[1], target_var[0].float())
            img_loss = criterion[1](outputs[2], target_var[1])
            rec_loss = criterion[1](outputs[3], target_var[2])
            # combined loss
            loss =  opts.cos_weight * cos_loss +\
                    opts.cls_weight * img_loss +\
                    opts.cls_weight * rec_loss 

            # measure performance and record losses
            cos_losses.update(cos_loss.data, inputs[0].size(0))
            img_losses.update(img_loss.data, inputs[0].size(0))
            rec_losses.update(rec_loss.data, inputs[0].size(0))
        else:
            loss = criterion(outputs[0], outputs[1], target_var[0])
            # measure performance and record loss
            cos_losses.update(loss.data[0], inputs[0].size(0))
        
        optimizer.zero_grad()               # 勾配の初期化
        loss.backward()                     # 勾配の計算
        optimizer.step()                    # パラメータの更新
        pbar.update(1)
    pbar.close()
    if opts.semantic_reg:
        print('Epoch: {0}\t'
                  'cos loss:{cos_loss.val:.4f} ({cos_loss.avg:.4f}) '
                  'img Loss:{img_loss.val:.4f} ({img_loss.avg:.4f}) '
                  'rec loss:{rec_loss.val:.4f} ({rec_loss.avg:.4f}) '
                  'vision_lr:({visionLR})-recipe_lr:({recipeLR}) '.format(
                   epoch, cos_loss=cos_losses, img_loss=img_losses,
                   rec_loss=rec_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr']))
    else:
         print('Epoch: {0}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
                   epoch, loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr']))  
    return cos_losses.val, img_losses.val, rec_losses.val

'''
検証ループ定義
'''
# 検証用関数(1epoch)
def one_valid(loader, model, criterion):
    print("LOG : validation phase")
    model.eval()           # モデルを推論モードに設定
    correct = 0            # 正解率計算用の変数を宣言
    total_loss = 0.0       # 1epochの損失合計
    data_num  = len(loader.dataset)              # テストデータの総数
    pbar = tqdm(total=int(data_num/BATCH_SIZE))  # プログレスバー設定
    with torch.no_grad():  # 推論時には勾配は不要(メモリ節約)
        for i, (inputs, targets) in enumerate(loader):
            # データをdeviceに載せる (image, inst, len(inst), ingr, len(ingr)), [target, img_id, rec_id]
            input_var  = [data.to(DEVICE, non_blocking=True) for data in inputs]
            target_var = [data.to(DEVICE, non_blocking=True) for data in targets[:-2]]
            outputs = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4]) # モデルから出力を得る 

            if i==0:
                data0 = outputs[0].data.cpu().numpy()
                data1 = outputs[1].data.cpu().numpy()
                data2 = targets[-2]
                data3 = targets[-1]
            else:
                data0 = np.concatenate((data0,outputs[0].data.cpu().numpy()),axis=0)
                data1 = np.concatenate((data1,outputs[1].data.cpu().numpy()),axis=0)
                data2 = np.concatenate((data2,targets[-2]),axis=0)
                data3 = np.concatenate((data3,targets[-1]),axis=0)
            pbar.update(1)
    pbar.close()
    medR, recall = rank(opts, data0, data1, data2) # img_embeds, rec_embeds, rec_ids
    print('Val medR:{medR:.4f}' ' Recall:{recall}'.format(medR=medR, recall=recall))
   
    return (medR, recall) # 各バッチごとのlossの平均

'''
その他定義
'''
def rank(opts, img_embeds, rec_embeds, rec_ids):
    random.seed(opts.seed)
    type_embedding = opts.embtype # default : "image"
    im_vecs        = img_embeds   # data0
    instr_vecs     = rec_embeds   # data1
    names          = rec_ids      # data2
    # Sort based on names to always pick same samples for medr
    idxs  = np.argsort(names)
    names = names[idxs]
    # Ranker
    N = opts.medr
    idxs = range(N)
    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):
        ids = random.sample(range(0,len(names)), N)
        im_sub    = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]
        ids_sub   = names[ids]
        # if params.embedding == 'image':
        if type_embedding == 'image':
            sims = np.dot(im_sub,instr_sub.T) # for im2recipe
        else:
            sims = np.dot(instr_sub,im_sub.T) # for recipe2im
        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}
        for ii in idxs:
            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii,:]
            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()
            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)
            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1
            # store the position
            med_rank.append(pos+1)
        for i in recall.keys():
            recall[i]=recall[i]/N
        med = np.median(med_rank)
        # print "median", med
        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)
    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10
    return np.average(glob_rank), glob_recall


# modelのパラメータ保存はval_lossがよくなったときだけ
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = CHECKPOINT + 'epoch%03d_val%.3f.pth.tar' % (state['epoch'],state['best_val']) 
    if is_best:
        torch.save(state, filename)

# 平均と現在の値を計算して保存するクラス
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 学習率lrの適用
def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
    optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
    # parameters corresponding to visionMLP 
    optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision 

    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial vision lr: %f' % optimizer.param_groups[1]['lr'])

    # after first modality change we set patience to 3 : d3
    PATIENCE = 2

'''
学習開始
'''
print("LOG : 未学習時のモデル性能検証...")
# 未学習時のモデルの性能の検証
valid_result = one_valid(validloader,  model, criterion)

# 学習開始
print("\n\n")
print("LOG : 学習開始...epoch", EPOCH, "まで")
for epoch in range(START_EPOCH, EPOCH+1): 
    cos_loss, img_loss, rec_loss = one_train(trainloader, model, criterion, optimizer, epoch)
    valid_medr, valid_recall = one_valid(validloader, model, criterion) 
    write_tbx(epoch, cos_loss, img_loss, rec_loss, valid_medr, valid_recall)
    
    # val_lossがよくならないのがpatience回続いたら学習するモダリティの変更
    if valid_medr >= best_val:  valtrack += 1
    else: valtrack = 0

    if valtrack >= PATIENCE:
        # we switch modalities
        opts.freeVision = opts.freeRecipe; opts.freeRecipe = not(opts.freeVision)
        # change the learning rate accordingly
        adjust_learning_rate(optimizer, epoch, opts) 
        valtrack = 0
    
    # best modelの保存
    is_best = valid_medr < best_val      # ロスが小さくなったか
    best_val = min(valid_medr, best_val) # 最高値の更新
    save_checkpoint({  # modelの保存
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_val': best_val,
        'optimizer_state_dict': optimizer.state_dict(),
        'valtrack': valtrack,
        'freeVision': opts.freeVision,
        'valid_medr': valid_medr
        }, is_best)
    print('Validation: %f (best) - %d (valtrack)' % (best_val, valtrack))
    print("- - - - - - - - - - - - - - - - - - - - -")

'''
後処理
'''
writer.close() # tbxをclose

'''
グラフは以下で確認可能
tensorboard --logdir="./tbX"
'''

print("LOG : Finish!!")