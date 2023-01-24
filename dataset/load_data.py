import csv
import os
import random
import PIL.Image as Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset.randaugment import RandAugment
import numpy as np
from dataset.augment import RandAugment_aug

label_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_test_dataset(img_dir):
    '''加载dfu_test的dataset数据, 并且生成迭代器'''
    # 获取图片路径
    imgs = []
    filename = os.listdir(img_dir)
    for i in filename:
        img_path = os.path.join(img_dir, i)
        imgs.append([img_path, []])
    return imgs

def load_train_dataset(cfg, name=''):
    '''
        顺序读图片文件路径以及标签
        return:
            label_imgs:[(img_path, label), ...]
            unlabel_imgs:[img_path, ...]
    '''

    img_dir = cfg.DATASET.IMG_ROOT
    label_dir = cfg.DATASET.LABEL_ROOT
    dic = {'label':[], 'unlabel':[]}
    f = open(label_dir, 'r')
    reader = csv.DictReader(f)  # 以字典形式读取
    # 寻找标签
    for row in reader:
        img_path = os.path.join(img_dir, row['image'])
        if row['image']<'4' or row['image']>'5':
            index = list(row.values()).index('1')
            if list(row.keys())[index] == 'none' or list(row.keys())[index] == 'infection':
                dic['label'].append([img_path, [0, index-1]])   # index 1, 2 变成0, 1
            else:
                dic['label'].append([img_path, [1, index-1]])   # index 3, 4 变成2, 3
        else:
            dic['unlabel'].append([img_path, [-1, -1]])
    f.close()
    random.shuffle(dic['label'])
    random.shuffle(dic['unlabel'])

    return dic['label'], dic['unlabel']

def load_augment_dataset(augment_dir,l=None):
    list = [] if l == None else l
    filename = os.listdir(augment_dir)
    for i in filename:
        img_path = os.path.join(augment_dir, i)
        list.append([img_path, [-1, -1]])
    return list

def generate_loader(imgs, size, islabel, name=''):
    # 生成有无标签的训练dataset类对象
    # 此时数据样本为[(tensor(C,H,W),int(class_id)), (tensor,int), ... ]
    if islabel is True:   # fixmatch and flexmatch 验证，train, test
        dataset = TDataset(imgs=imgs, transform=label_transform)
        # dataset = TDataset(imgs=imgs, transform=Transform_AUG())
    else:                   # fixmatch 的无标签
        dataset = TDataset(imgs=imgs, transform=TransformFixMatch())    
    # 生成dataset类的迭代对象
    # 在这一步数据变成[[tensor(batch_size, C, W ,H), tensor(class_id)], ...]
    # shuffle=True 每个epoch对数据重新排序
    loader = DataLoader(dataset, batch_size=size, shuffle=True) #  replacement 可以重复采样
    return loader

class Transform_AUG(object):
    def __init__(self):
        self.strong = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            RandAugment_aug(n=1, m=13),
            # RandAugment(n=1, m=13)
            ])
        self.nomalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        strong = self.strong(x)
        return self.nomalize(strong)

class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandAugment(n=2, m=10)
            # RandAugment_aug(n=3, m=13)
            ])
        self.nomalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.nomalize(weak), self.nomalize(strong)

class TDataset(Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]  # list类型
        img = Image.open(img_path)
        img = self.transform(img)   # 无标签的图片会返回一个[weak, strong]
        # name = img_path.split('/')[-1]
        return img_path, index, img, label

    def __len__(self):
        return len(self.imgs)

def get_kfold_dataset(k, i, label_imgs):
    """
        返回第 i+1 折 (i = 0 -> k-1)
        交叉验证时所需要的训练和验证数据, X_train为训练集, X_valid为验证集
    """
    if k==1:
        return label_imgs, label_imgs
    fold_size = len(label_imgs) // k  # 每份的个数:数据总条数/折数（组数）
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        valid = label_imgs[val_start:val_end]
        train = label_imgs[0:val_start]+label_imgs[val_end:]
    else:  # 若是最后一折交叉验证
        valid = label_imgs[val_start:]
        train = label_imgs[0:val_start]
    False_rec = [x for x in valid if x in train]
    print("error") if len(False_rec)!=0 else print("currect")
    return train, valid

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

