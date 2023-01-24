import argparse
import csv
import warnings
import torch
from configs import cfg, update_config

from dataset import generate_loader, get_kfold_dataset, load_train_dataset
from train import (BasicTrain, FixMatch, test)
from utils import create_logger
from utils.ensemble import result_avg

warnings.filterwarnings('ignore')

# 参数设置
def parse_args():
    parser = argparse.ArgumentParser(description='dfu')
    parser.add_argument(
        '--cfg',                                                        
        required=False,
        default="./dfu/configs/Basic.EfficientNet.yaml"
    )
    args = parser.parse_args()
    return args
 
def step(cfg, logger, model_save_file, train_imgs, valide_imgs, unlabel_imgs, k, step):
    ssl_type = eval('cfg.TRAIN_'+ str(step) + '.SSL_TYPE')
    mix_unlabel_and_label = False
    train_imgs_u_and_l = train_imgs.copy()

    # k折提取数据
    loader = []
    train_loader = generate_loader(train_imgs_u_and_l, cfg.DATASET.LABEL_BATCHSIZE, True, ssl_type)
    loader.append(train_loader)
    valide_loader = generate_loader(valide_imgs, cfg.DATASET.LABEL_BATCHSIZE, True)
    loader.append(valide_loader)
    if unlabel_imgs!=[]:   # 前者说明 
        un_loader = generate_loader(unlabel_imgs, cfg.DATASET.UNLABEL_BATCHSIZE, False, ssl_type)
        loader.append(un_loader)
    else:
        loader.append(None)
    
    if unlabel_imgs==[] and mix_unlabel_and_label == False:
        t = BasicTrain(cfg, loader, logger, model_save_file, k, step)
    else:
        t = eval(ssl_type)(cfg, loader, logger, model_save_file, k, step)

    t.train()

def main(cfg, logger, model_save_file):
    start_k = 0
    label_imgs, unlabel_imgs = load_train_dataset(cfg)

    logger.info(
        "Finish, total number:{}, label number:{}, unlabel number:{}".format(
            len(label_imgs)+len(unlabel_imgs), len(label_imgs), len(unlabel_imgs)))
    
    # 开始五折交叉验证训练
    logger.info("Start training")
    for k in range(start_k, cfg.BASIC.K):
        logger.info('The {} is converted into a validation set'.format(k))
        train_imgs, valide_imgs = get_kfold_dataset(cfg.BASIC.K, k, label_imgs)
        logger.info(
            "train_imgs:{}, valide image:{}".format(
                len(train_imgs), len(valide_imgs)))
        """
            第一阶段的训练开始, infection和control算0类, ischaemia和both算1类
            0类有5107, 1类有848。6:1
        """
        if cfg.BASIC.STEP == 0:
            step(cfg, logger, model_save_file, train_imgs, valide_imgs, unlabel_imgs, k, 0)
            test(k, cfg, logger, model_save_file)
            torch.cuda.empty_cache()
        else:
            pass

if __name__ == '__main__':
    args = parse_args()
    
    update_config(cfg, args.cfg)                    
    logger, log_dir= create_logger(cfg)
    main(cfg, logger, log_dir)
    result_avg(log_dir)

