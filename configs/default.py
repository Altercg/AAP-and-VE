from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.BASIC = CN()
_C.BASIC.NAME = ''
_C.BASIC.LOG_DIR = ''

_C.BASIC.K = 5                                      # k折交叉验证
_C.BASIC.CLASS_NUM = 4
_C.BASIC.VALID_STEP = 10
_C.BASIC.STEP = 0

_C.BASIC.RESUME = False
_C.BASIC.RESUME_PATH = ''
_C.BASIC.RESUME_DATA = ''

# ----- DATASET BUILDER -----
_C.DATASET = CN()
_C.DATASET.IMG_ROOT = ''
_C.DATASET.LABEL_ROOT = ''
_C.DATASET.LABEL_BATCHSIZE = 25
_C.DATASET.UNLABEL_BATCHSIZE = 21
_C.DATASET.AUG = ''

# ----- TRAIN BUILDER -----
_C.TRAIN_0 = CN()
_C.TRAIN_0.EPOCH = 90
_C.TRAIN_0.NET_TYPE = ''
_C.TRAIN_0.SSL_TYPE = ''
_C.TRAIN_0.THRESH_WARMUP = False
_C.TRAIN_0.THRESHOLD = 0.95   # 伪标签置信度
_C.TRAIN_0.DROUPOUT = 0.0

_C.TRAIN_0.LOSS = CN()
_C.TRAIN_0.LOSS.LOSS_TYPE = ''
_C.TRAIN_0.LOSS.WEIGHT = False

_C.TRAIN_0.OPTIMIZER = CN()
_C.TRAIN_0.OPTIMIZER.TYPE = ''
_C.TRAIN_0.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN_0.OPTIMIZER.WEIGHT_DECAY = 0.0
_C.TRAIN_0.OPTIMIZER.LR_SCHEDULER = 0.0001

_C.TRAIN_0.SCHEDULER = CN()
_C.TRAIN_0.SCHEDULER.TYPE = ''
_C.TRAIN_0.SCHEDULER.PARAMETER_1 = None
_C.TRAIN_0.SCHEDULER.PARAMETER_2 = None

# ----- TEST BUILDER -----
_C.TEST = CN()
_C.TEST.IMG_ROOT = ''
_C.TEST.BATCHSIZE = 128

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args)
    cfg.freeze()