import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.utils import get_model, get_optimizer, get_scheduler, save_all, loss_drawing
from utils import evaluating_indicator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FixMatch:
    def __init__(self, cfg, loader, logger, model_save_file, k, step):
        super(FixMatch, self).__init__()
        self.train_loader,  self.valide_loader, self.un_loader = loader
        self.model_save_file = model_save_file
        self.logger = logger
        self.step = step        # 这个存的是现阶段
        self.cfg = cfg          # 这里存的是从哪个阶段开始
        self.k = k
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        self.num_classes = cfg.BASIC.CLASS_NUM
        self.get_model_and_optimizer()

    def get_model_and_optimizer(self, resume=False, checkpoint=None):
        self.model = get_model(self.cfg, self.step)
        self.optimizer = get_optimizer(self.cfg, self.model, self.step)
        self.scheduler = get_scheduler(self.cfg, self.optimizer, self.step)
        self.start_epoch = 0
        if resume == True:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            # self.start_epoch = checkpoint['epoch']
        self.model.to(device)

    def train(self):
        best_f1 = (0, 0)
        # all_loss = []
        # all_f1 = []

        threshold = eval('self.cfg.TRAIN_'+ str(self.step) + '.THRESHOLD')
        epoch = eval('self.cfg.TRAIN_'+ str(self.step) + '.EPOCH')
        thresh_warmup = eval('self.cfg.TRAIN_'+ str(self.step) + '.THRESH_WARMUP')

        # 用来无标签损失暖机的
        epoch_iter = len(self.train_loader)
        num_train_iter = epoch * epoch_iter
        num_iter = 0
        unsup_warmup = 1

        for e in range(self.start_epoch, epoch):
            loss_e = 0
            num_use = 0
            self.model.train()
            for i, (input_l, input_u) in enumerate(zip(self.train_loader, self.un_loader)):
                if thresh_warmup == True:
                    unsup_warmup = np.clip(num_iter/(0.3 * num_train_iter), a_min=0.0, a_max=1.0)
                    unsup_warmup = torch.tensor(unsup_warmup)
                    
                # 提取有标签的训练数据 torch.Size([b, 3, 224, 224]) torch.Size([b])
                _, _, x, label = input_l
                if self.step == 1:  # 1代表第一阶段
                    y = label[0].type(torch.LongTensor)
                elif self.step == 2:
                    y = label[1].type(torch.LongTensor)
                elif self.step == 3:
                    y = label[1].type(torch.LongTensor)
                    y -= 2
                else:
                    y = label[1].type(torch.LongTensor)
                y = y.to(device)
                label_batchsize = y.shape[0]
                # 无标签 torch.Size([ub, 3, 224, 224]) torch.Size([ub, 3, 224, 224])
                _, _, x_u, _ = input_u
                w, s = x_u
                # 数据合并 torch.Size([b+ub+ub, 3, 224, 224])
                inputs = torch.cat((x, w, s), dim=0).to(device)
                outputs = self.model(inputs)
                # 数据拆解
                # torch.Size([b, 4]) torch.Size([ub, 4]) torch.Size([ub, 4])
                out_x = outputs[:label_batchsize]
                out_w, out_s = outputs[label_batchsize:].chunk(2)

                # 有标签损失
                # 这个函数包含了softmax, weight可以改别不同类的权重，适合训练样本不均衡
                # if self.cfg.TRAIN.LOSS.LOSS_TYPE == 'CrossEntropy' and  self.cfg.TRAIN.LOSS.WEIGHT:
                #     if self.step == 1:
                #         weight = (torch.from_numpy(np.array([1, 6])).float()).to(device)
                #     elif self.step == 2:
                #         weight = (torch.from_numpy(np.array([1, 1])).float()).to(device)
                #     elif self.step == 3:
                #         weight = (torch.from_numpy(np.array([3, 1])).float()).to(device)
                #     loss_x = nn.CrossEntropyLoss(weight=weight)(out_x, y)
                #elif self.cfg.TRAIN.LOSS.LOSS_TYPE == 'CrossEntropy':
                #    loss_x = nn.CrossEntropyLoss()(out_x, y)

                loss_x = nn.CrossEntropyLoss()(out_x, y)
                # 无标签损失
                # 先获得弱标签的伪标签
                pseudo_label = torch.softmax(out_w.detach(), dim=1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(threshold).float()
                num_use += torch.count_nonzero(mask)
                # 比较损失
                loss_w = (F.cross_entropy(out_s, targets_u, reduction='none')*mask).mean()

                # 参数更新
                loss = unsup_warmup * loss_w + loss_x
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_e += loss.detach().cpu()
                num_iter += 1
                # self.scheduler.step(e+i/epoch_iter)

            if (e+1) % self.cfg.BASIC.VALID_STEP == 0 :
                all = self.validate()
                f1_score = all['macro avg']['f1-score']
                # all_f1.append(f1_score)
                # 保存中间模型   
                if best_f1[1] < f1_score: # 存储该折最好模型
                    best_f1 = (e+1, f1_score)
                    torch.save({
                        'model_state_dict': self.model.state_dict()
                    }, self.model_save_file+'{}-{}-best-model.pt'.format(self.k, self.step))

            info = "E|L:{}|{} lr:{} best-f1:{} num_use:{}".format(
                e+1, loss_e, self.optimizer.state_dict()['param_groups'][0]['lr'], best_f1, num_use)
            self.logger.info(info)

            if (e+1) % self.cfg.BASIC.VALID_STEP == 0:
                save_all(self.logger, all)
            self.scheduler.step()
            # all_loss.append(loss_e)
            
        # loss_drawing(all_loss, all_f1, self.k, self.model_save_file, self.step)

    def cal_unlabel_img(self):    
        # 修改dic unlabel, 第一阶段的值
        threshold = eval('self.cfg.TRAIN_'+ str(self.step) + '.THRESHOLD')
        self.model.eval()
        unlabel_imgs = []
        with torch.no_grad():   
            for i, j in enumerate(self.un_loader):
                name, _,  x, _ = j
                w,_ = x
                inputs = w.to(device)   # 这里是用增强的弱图分类，如果用原图呢？其实也不用，弱增强只是进行旋转而已
                outputs = self.model(inputs)
                out = torch.softmax(outputs, dim=1)
                max_probs, targets_u = torch.max(out, dim=-1)
                mask = max_probs.ge(threshold).float()
                for i, m in enumerate(mask):
                    if m==1:
                        unlabel_imgs.append([name[i], [targets_u[i], -1]])
            return unlabel_imgs      # 存在可能, 把所有的unlabel都属于了一个类别
 
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            y_true = None
            y_pred = None
            imgs_path = None
            for i, j in enumerate(self.valide_loader):
                img_path, _, x, label = j
                inputs = x.to(device)
                if self.step == 1:  # 1代表第一阶段
                    y = label[0].type(torch.LongTensor)
                elif self.step == 2:
                    y = label[1].type(torch.LongTensor)
                elif self.step == 3:
                    y = label[1].type(torch.LongTensor)
                    y -= 2
                else:
                    y = label[1].type(torch.LongTensor)
                outputs = self.model(inputs)
                out = torch.softmax(outputs, dim=1)
                pred = out.max(1, keepdim=False)[1].cpu()
                if i == 0:
                    y_true = y
                    y_pred = pred
                    imgs_path = img_path
                else:
                    y_true = torch.cat((y_true, y), dim=-1)
                    y_pred = torch.cat((y_pred, pred), dim=-1)
                    imgs_path += img_path
            all = evaluating_indicator(y_true, y_pred, self.step, self.model_save_file, imgs_path, self.k)
        return all
