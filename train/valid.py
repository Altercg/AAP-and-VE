import torch
from utils import evaluating_indicator

def validate(model, val_loader, device, model_save_file, k, step):
    model.eval()
    with torch.no_grad():
        y_true = None
        y_pred = None
        # imgs_path = []
        for i, j in enumerate(val_loader):
            _, _, x, label = j
            inputs = x.to(device)
            if step == 1:  # 1代表第一阶段
                y = label[0].type(torch.LongTensor)
            else:
                y = label[1].type(torch.LongTensor)
            outputs = model(inputs)
            out = torch.softmax(outputs, dim=1)
            pred = out.max(1, keepdim=False)[1].cpu()
            if i == 0:
                y_true = y
                y_pred = pred
            else:
                y_true = torch.cat((y_true, y), dim=-1)
                y_pred = torch.cat((y_pred, pred), dim=-1)
            # imgs_path += img_path
        all = evaluating_indicator(y_true, y_pred, step)
       
        return all
