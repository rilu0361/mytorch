"""
my_util.py
関数のまとめ
"""
import shutil, os

'''
ファイルの存在チェック
'''
def check_file(path):
    if os.path.exists(path):
        check = input("CHECK : すでに{}が存在しています。削除しますか？(y/n) : ".format(path))
        if check=="y" or check=="Y":
            shutil.rmtree(path)
            print("FILE : ",path,"を削除しました.")
        else:
            print("INFO : 任意の操作の後再実行してください。") 
            exit()

'''
tensorboard設定
'''
# tensorboardへの記録用関数
def write_tb(epoch, cos_loss, img_loss, rec_loss, val_medr, val_recall):
    print(epoch, cos_loss, img_loss, rec_loss, val_medr)
    writer.add_scalar('train/cos_loss', cos_loss, epoch) 
    writer.add_scalar('train/img_loss', img_loss, epoch)
    writer.add_scalar('train/rec_loss', rec_loss, epoch)
    writer.add_scalar('valid/medr',     val_medr, epoch)
    writer.add_scalar('valid/recall@1',  val_recall[1],  epoch)
    writer.add_scalar('valid/recall@5',  val_recall[5],  epoch)
    writer.add_scalar('valid/recall@10', val_recall[10], epoch)
