import torch
import csv
from dataset.load_data import load_test_dataset, generate_loader
from utils.utils import get_model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(k, cfg, logger, model_save_file):
    # 导入数据
    test_imgs = load_test_dataset(cfg.TEST.IMG_ROOT)
    test_loader = generate_loader(test_imgs, cfg.TEST.BATCHSIZE, True)
    logger.info("Finish, total number:{}".format(len(test_loader.dataset)))
    # 导入模型
    logger.info("-"*30+"Start test"+"-"*30)
    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg, 0)
    
    model.load_state_dict(torch.load(model_save_file +'{}-0-best-model.pt'.format(k))['model_state_dict'])
    # ----- END MODEL BUILDER -----
    model.to(device)
    model.eval()
    # 开始测试
    with torch.no_grad():
        result_filename =  model_save_file +'{}-result.csv'.format(k)
        # 输出结果的格式为cvs
        with open(result_filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                    ['image', 'none', 'infection', 'ischaemia', 'both'])
            for img in test_loader:
                image_path, _, x, _ = img
                inputs = x.to(device)
                outputs = model(inputs)
                # outputs = model(inputs, inputs)
                # out = torch.softmax(z_i, dim=1)
                outputs = torch.softmax(outputs, dim=1).cpu().numpy()
                np.set_printoptions(
                    precision=15, floatmode='fixed', suppress=True)
                outs = outputs.astype("float32").tolist()
                for i, o in enumerate(outs):
                    name = image_path[i].split('/')[-1]
                    o = ['{:.15f}'.format(round(i, 15)) for i in o]
                    o.insert(0, name)
                    writer.writerow(o)
        csvfile.close()
    logger.info("Finish!")
    