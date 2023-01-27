import csv
import torch
import numpy as np
from collections import Counter

def result_avg(path='./Desktop/'):
    ensemble_num = 5

    f0 = open(path+'0-result.csv', 'r')
    reader0 = csv.reader(f0)
    f1 = open(path+'1-result.csv', 'r')
    reader1 = csv.reader(f1)
    f2 = open(path+'2-result.csv', 'r')
    reader2 = csv.reader(f2)
    f3 = open(path+'3-result.csv', 'r')
    reader3 = csv.reader(f3)
    f4 = open(path+'4-result.csv', 'r')
    reader4 = csv.reader(f4)

    result = None
    # 数据导入
    result0 = []
    result1 = []
    result2 = []
    result3 = []
    result4 = []
    for q in range(0, ensemble_num):
        for i, row in enumerate(eval('reader'+str(q))):
            if i == 0:
                continue
            for j in range(1, 5):   # 把字符变数字
                row[j] = float(row[j])
            eval('result'+str(q)).append(row)
    # 排序
    for q in range(0, ensemble_num):
        eval('result'+str(q)).sort()

    # 去除图片名字
    r0 = []
    r1 = []
    r2 = []
    r3 = []
    r4 = []

    for q in range(0, ensemble_num):
        for i in range(len(eval('result'+str(q)))):
            # 存储元素
            eval('r'+str(q)).append(eval('result'+str(q))[i][1:])

    r0 = torch.tensor(r0) # torch.size(n, 4)  类别概率
    r1 = torch.tensor(r1)
    r2 = torch.tensor(r2)
    r3 = torch.tensor(r3)
    r4 = torch.tensor(r4)

    # 数据求和
    result = r0
    for q in range(1, ensemble_num):
        result += eval('r'+str(q))
    # 数据取平均
    result /= ensemble_num
    name = np.array(result0)
    name = name[:, 0].reshape(-1, 1)
    result = result.numpy()
    save(np.concatenate([name,result],axis=1), ['image', 'none', 'infection', 'ischaemia', 'both'], path+'avg_result.csv')

def result_vote_diff_model():
    """
        投票规则：
            1. 5个结果类别一致的情况, 存取概率最大的预测条目；
            2. 5个结果类别出现不一致的情况:
                (1). 有模型预测结果高于0.95, 力排众议; 
                (2). 多个高于0.95,并且各个类别不一致, 少数服从多数。
                (3). 并未出现力排众议的, 少数服从多数;
    """
    ensemble_num = 3
    path1 = './avg_result-b.csv'
    path2 = './avg_result-d.csv'
    path3 = './avg_result-e.csv'
    f0 = open(path1, 'r')
    reader0 = csv.reader(f0)
    f1 = open(path2, 'r')
    reader1 = csv.reader(f1)
    f2 = open(path3, 'r')
    reader2 = csv.reader(f2)

    result = []
    # 数据导入
    result0 = []
    result1 = []
    result2 = []
    for q in range(0, ensemble_num):
        for i, row in enumerate(eval('reader'+str(q))):
            if i == 0:
                continue
            for j in range(1, 5):
                row[j] = float(row[j])
            eval('result'+str(q)).append(row)
    
    # 排序
    for q in range(0, ensemble_num):
        eval('result'+str(q)).sort()

    # 去除图片名字
    r0 = []
    r1 = []
    r2 = []

    for q in range(0, ensemble_num):
        for i in range(len(eval('result'+str(q)))):
            # 存储元素
            eval('r'+str(q)).append(eval('result'+str(q))[i][1:])

    r0 = torch.tensor(r0) # torch.size(n, 4)  类别概率
    r1 = torch.tensor(r1)
    r2 = torch.tensor(r2)

    # 每个模型的每张图片的最大值以及类别
    all_probs = None
    all_targets = None
    for q in range(0, ensemble_num):
        probs, targets = torch.max(eval('r'+str(q)), dim=-1)
        if q == 0:
            all_probs = probs.view(-1, 1)   # torch.size(n, 1)
            all_targets = targets.view(-1, 1)
        else:
            all_probs = torch.cat((all_probs, probs.view(-1, 1)), dim=-1)  # torch.size(n, 3)
            all_targets = torch.cat((all_targets, targets.view(-1, 1)), dim=-1)  # torch.size(n, 3)
    
    # 统计每张图片的类别个数
    count_dis_prob = 0
    max_dis_prob = 0
    count=0
    for i in range(len(result0)):
        count_targets = Counter(all_targets[i].tolist())
        prob, model_result = torch.max(all_probs[i], dim=-1)    # 概率以及应该选哪个模型
        # 查看类别是否一致
        if len(count_targets.keys()) == 1: # 类别一致
            # 直接保存最大类别的
            pass
        else:
            # 少数服从多数, 多数中最大的
            # 有可能这个最大值是少数派的不超过0.95的
            target = max(count_targets, key=count_targets.get)   # 获得最大值的键，得到图片应该属于什么类别->得到多数类
            # 多数类中是否有超过0.95的存在？
            # print(target)
            mask = (all_targets[i]==target)
            mask = mask.type(torch.FloatTensor)
            max_prob, model_r = torch.max(mask*all_probs[i], dim=-1)   # 获得多数类的最大值
            # print(prob, max_prob)
            if model_r == model_result or max_prob >= 0.95: # 最大值出现在了多数类里面, 多数类也有大于0.95的值，虽然比最大值小
                # max_prob >= 0.95 or prob < 0.95:
                model_result = model_r

            mask = mask*all_probs[i]
            mask = mask.numpy()
            min_prob = np.min(np.setdiff1d(mask, [0]))
            count_dis_prob += max_prob-min_prob
            count += 1
            if(max_prob-min_prob) > max_dis_prob:
                max_dis_prob = max_prob-min_prob
            
        result.append(eval('result'+str(model_result.item()))[i])

    save(result, ['image', 'none', 'infection', 'ischaemia', 'both'], './vote_diff_result.csv')
    # save(all_probs, ['bit','densnet', 'efficientnet'], './diff_all_probs.csv')
    # save(all_targets, ['bit','densnet', 'efficientnet'], './diff_all_targets.csv')
    # print(max_dis_prob, count_dis_prob, count, count_dis_prob/count)

    f0.close()
    f1.close()
    f2.close()

def result_vote_model():
    """
        投票规则：
            1. 5个结果类别一致的情况, 存取概率最大的预测条目；
            2. 5个结果类别出现不一致的情况: 少数服从多数;
    """
    ensemble_num = 3
    path1 = './avg_result-b.csv'
    path2 = './avg_result-d.csv'
    path3 = './avg_result-e.csv'
    f0 = open(path1, 'r')
    reader0 = csv.reader(f0)
    f1 = open(path2, 'r')
    reader1 = csv.reader(f1)
    f2 = open(path3, 'r')
    reader2 = csv.reader(f2)

    result = []
    # 数据导入
    result0 = []
    result1 = []
    result2 = []
    for q in range(0, ensemble_num):
        for i, row in enumerate(eval('reader'+str(q))):
            if i == 0:
                continue
            for j in range(1, 5):
                row[j] = float(row[j])
            eval('result'+str(q)).append(row)
    
    # 排序
    for q in range(0, ensemble_num):
        eval('result'+str(q)).sort()

    # 去除图片名字
    r0 = []
    r1 = []
    r2 = []

    for q in range(0, ensemble_num):
        for i in range(len(eval('result'+str(q)))):
            # 存储元素
            eval('r'+str(q)).append(eval('result'+str(q))[i][1:])

    r0 = torch.tensor(r0) # torch.size(n, 4)  类别概率
    r1 = torch.tensor(r1)
    r2 = torch.tensor(r2)

    # 每个模型的每张图片的最大值以及类别
    all_probs = None
    all_targets = None
    for q in range(0, ensemble_num):
        probs, targets = torch.max(eval('r'+str(q)), dim=-1)
        if q == 0:
            all_probs = probs.view(-1, 1)   # torch.size(n, 1)
            all_targets = targets.view(-1, 1)
        else:
            all_probs = torch.cat((all_probs, probs.view(-1, 1)), dim=-1)  # torch.size(n, 3)
            all_targets = torch.cat((all_targets, targets.view(-1, 1)), dim=-1)  # torch.size(n, 3)
    
    # 统计每张图片的类别个数
    count_dis_prob = 0
    max_dis_prob = 0
    count=0
    for i in range(len(result0)):
        count_targets = Counter(all_targets[i].tolist())
        prob, model_result = torch.max(all_probs[i], dim=-1)    # 概率以及应该选哪个模型
        # 查看类别是否一致
        if len(count_targets.keys()) == 1: # 类别一致
            # 直接保存最大类别的
            pass
        else:
            # 少数服从多数, 多数中最大的
            # 有可能这个最大值是少数派的不超过0.95的
            target = max(count_targets, key=count_targets.get)   # 获得最大值的键，得到图片应该属于什么类别->得到多数类
            mask = (all_targets[i]==target)
            mask = mask.type(torch.FloatTensor)
            max_prob, model_r = torch.max(mask*all_probs[i], dim=-1)   # 获得多数类的最大值
            model_result = model_r
            mask = mask*all_probs[i]
            mask = mask.numpy()
            min_prob = np.min(np.setdiff1d(mask, [0]))
            count_dis_prob += max_prob-min_prob
            count += 1
            if(max_prob-min_prob) > max_dis_prob:
                max_dis_prob = max_prob-min_prob
            
        result.append(eval('result'+str(model_result.item()))[i])

    save(result, ['image', 'none', 'infection', 'ischaemia', 'both'], './vote_result.csv')
    # save(all_probs, ['bit','densnet', 'efficientnet'], './diff_all_probs.csv')
    # save(all_targets, ['bit','densnet', 'efficientnet'], './diff_all_targets.csv')
    # print(max_dis_prob, count_dis_prob, count, count_dis_prob/count)

    f0.close()
    f1.close()
    f2.close()

def save(result, row_name, file_path='./result.csv'):
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(row_name)
        for i, o in enumerate(result):
            writer.writerow(o)

if __name__ == "__main__":
    result_avg()
    result_vote_diff_model()   # 投票取最大条目，还是投票取类别呢？ 一样的
    result_vote_model()
