import datetime
import numpy as np
import logging
import os
import csv
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.efficientnet_pytorch import EfficientNet
from utils.bit_pytorch import models as BiT
from torchvision.models import DenseNet

def create_logger(cfg):
    config_name = cfg.BASIC.NAME
    log_dir = cfg.BASIC.LOG_DIR
    time_str = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
    log_dir = log_dir+config_name+time_str+'/'  # 文件夹路径
    os.mkdir(log_dir)                       # 创建文件夹
    log_name = "{}_{}.log".format(config_name, time_str)
    log_file = str(os.path.join(log_dir, log_name)) # log文件
    head = "%(asctime)s %(message)s" # 输出格式
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format=head)
    logger = logging.getLogger('DFU')    # 实例一个日志器
    console = logging.StreamHandler()   # 设置处理器讲日志输出
    logging.getLogger('DFU').addHandler(console)
    logger.info("-----------------Cfg is set as follow---------------")
    logger.info(cfg)
    logger.info("----------------------------------------------------")
    return logger, log_dir

def get_model(cfg, step):
    class_num = cfg.BASIC.CLASS_NUM
    net_type = eval('cfg.TRAIN_'+str(step)+'.NET_TYPE')
    ssl_type = eval('cfg.TRAIN_'+ str(step) + '.SSL_TYPE')
    dropout_rate= eval('cfg.TRAIN_'+str(step)+'.DROUPOUT')

    if net_type == 'EfficientNetb2':
        model = EfficientNet.from_name('efficientnet-b2', dropout_rate=dropout_rate)
        model.load_state_dict(torch.load("./dfu/utils/efficientnet_pytorch/efficientnet-b2.pth"))
        in_f = model._fc.in_features  # 改变类别
        model._fc = nn.Linear(in_f, class_num, bias=True)
    elif net_type == 'DenseNet201':
        model = models.densenet201(pretrained=True, drop_rate=dropout_rate)
        in_f = model.classifier.in_features  # 改变类别
        model.classifier = nn.Linear(in_f, class_num, bias=True)
    elif net_type == 'BiT101':
        model = BiT.KNOWN_MODELS['BiT-M-R101x1'](head_size=class_num, zero_head=True)
        model.load_from(np.load(f"./dfu/utils/bit_pytorch/BiT-M-R101x1.npz"))
    return model

def get_optimizer(cfg, model, step):
    lr = eval('cfg.TRAIN_'+str(step)+'.OPTIMIZER.LR_SCHEDULER')
    momentum = eval('cfg.TRAIN_'+str(step)+'.OPTIMIZER.MOMENTUM')
    weight_decay = eval('cfg.TRAIN_'+str(step)+'.OPTIMIZER.WEIGHT_DECAY')
    opti_type = eval('cfg.TRAIN_'+str(step)+'.OPTIMIZER.TYPE')

    if opti_type == 'Adam': # pimodel 和 temporal 用
        optimizer = Adam(model.parameters(), lr)# self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, 
    else:
        optimizer = eval(opti_type)(
            model.parameters(),
            momentum=momentum, 
            lr=lr,
            weight_decay=weight_decay
        )
    return optimizer

def get_scheduler(cfg, optimizer, step):
    sche_type = eval('cfg.TRAIN_'+str(step)+'.SCHEDULER.TYPE')
    paramenter_1 = eval('cfg.TRAIN_'+str(step)+'.SCHEDULER.PARAMETER_1')
    paramenter_2 = eval('cfg.TRAIN_'+str(step)+'.SCHEDULER.PARAMETER_2')

    scheduler = eval(sche_type)(optimizer, paramenter_1, paramenter_2)
    return scheduler

def evaluating_indicator(y_true, y_pred, step, *args):
    """
        return: 'per_class_indicator' include per class precision, recall,
                    f1-score, support(在真值里面出现的次数),
                    per indicator macro-avg
                micro-average f1-score
                micro-average auc
    """
    all = classification_report(
        y_true=y_true, y_pred=y_pred,
        labels=[0, 1, 2, 3],
        target_names=["none", "infection", "ischaemia",  "both"],
        output_dict=True)    # 可以转化为字典，使用指标数值
    # 输出结果的格式为cvs
    if len(args)!= 0 and isinstance(args[0], str):
        error_file = args[0]+ str(args[2])+ '-' + str(step) +'error_file'
        with open(error_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['image', 'true', 'pred'])
            for i, t in enumerate(zip(y_true, y_pred)):
                true, pred = t
                if true != pred:
                    img = args[1][i].split('/')[-1]
                    error_img = [img, int(true), int(pred)]
                    writer.writerow(error_img)
        csvfile.close()
    return all

def loss_drawing(all_loss, all_f1, k, model_save_path, step):
    epoch = range(1, len(all_loss)+1)
    epoch2= range(step, len(all_f1)*step+1, step)
    plt.plot(epoch, all_loss, 'b', label='Train_loss')
    plt.plot(epoch2, all_f1, 'r', label='F1-score')
    plt.legend()
    plt.savefig(model_save_path + "{}_{}_loss_f1-score.png".format(k, step))
    plt.clf()

def save_all(logger, all):
    for i in all.keys():
        logger.info("{}:{}".format(i, all[i]))
