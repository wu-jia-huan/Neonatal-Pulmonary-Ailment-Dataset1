import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))#生成了数据的索引的随机排列
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    # shuffled_target = [target[i].item() for i in indices]
    # shuffled_target_tensor = torch.tensor(shuffled_target).to(data.device)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    # Compute output
    target_a = target
    target_b = shuffled_target
    return data, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


eps = 1e-7

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels.long()) #zzy修改.long()

        # RCE
        pred = F.softmax(pred, dim=1) #softmax函数将原始分数转换为标准化的概率分布，使得概率之和为1
        pred = torch.clamp(pred, min=1e-7, max=1.0) #clamp将张量pred中的所有元素裁剪到指定的取值范围内
        label_one_hot = torch.nn.functional.one_hot(labels.to(torch.int64), self.num_classes).float().to(self.device) #zzy修改to(torch.int64)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
