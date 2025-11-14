import argparse
import logging
import os
import random
import re
import shutil
import sys
import time
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset_yrh import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_volume_classfier, test_single_volume_classfier_Xi_Ru
from dataloaders.dataset_yrh import BaseDataSets, RandomGenerator, BaseDataSets_xueyoubing, RandomGenerator_single, BaseDataSets_Xi_Ru_txt
from cutmix.cutmix import cutmix, SCELoss
from networks.resnet import ResNet18,ResNet34

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='DenseNet_2_class', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,# 吸入综合征的是二分类
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')


# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--text_file',type=str,default='/mnt/sdd/wjh/Fei_Bing/dataset_text/Xi_Ru.csv',help='text-file')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 离散化函数
def discretize_age(age_str):
    if '天' in age_str:
        # 提取天数并转换为小时
        days = int(age_str.replace('天', ''))
        hours = days * 24
    else:
        # 提取小时
        hours = int(age_str.replace('小时', ''))
    return hours // 24  # 每24小时为一个区间


def discretize_gestational_age(gestation_str):
    # 去掉"周"以及其他可能存在的空格
    week_str = gestation_str.replace('周', '').replace(' ', '')
    
    if '+' in week_str:
        # 处理形如 '37+6' 的情况
        weeks = int(week_str.split('+')[0])  # 只提取周数
    else:
        # 处理形如 '40' 或者 '40周' 的情况
        if week_str.isdigit():
            weeks = int(week_str)
        else:
            weeks = 26  # 如果格式不正确，返回默认值
    index = (weeks - 26)//5
    # print("index",index)
    return index # 假设26周是最小值


def discretize_birth_weight(weight_str):
    weight = int(weight_str.replace('g', ''))
    return weight // 1000  # 每1000g为一个区间

# 定义类别映射
gender_mapping = {'女': 0, '男': 1}
delivery_mapping = {'顺产': 0, '剖宫产': 1,'臀位助产':0}
asphyxia_mapping = {'否': 0, '是': 1,'无':0}

# Embedding层
embedding_dim = 20  # 嵌入维度
gender_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
delivery_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
asphyxia_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)

# 数值数据的Embedding
age_embedding = nn.Embedding(num_embeddings=22, embedding_dim=embedding_dim)  # 最大的22天
gestational_age_embedding = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)  # 假设26周到45周
birth_weight_embedding = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)  # 假设最多5个区间

# 获取每张图像对应的文本数据并向量化的函数
def vectorize_text_info(text_info):
    # 去掉 '是否窒息' 列中的 '无'
    asphyxia_list = [asphyxia if asphyxia != '无' else '否' for asphyxia in text_info['是否窒息']]

    delivery_idx = torch.tensor([
    delivery_mapping[mode] if pd.notna(mode) else 0  # 如果是 NaN，则默认映射为 0
    for mode in text_info['生产方式']
])
    gender_idx = torch.tensor([gender_mapping[gen] for gen in text_info['性别']])
    asphyxia_idx = torch.tensor([asphyxia_mapping[asphyxia] for asphyxia in asphyxia_list])

    age_idx = torch.tensor([discretize_age(age) for age in text_info['年龄']])
    gestational_age_idx = torch.tensor([discretize_gestational_age(gest) for gest in text_info['胎龄']])
    birth_weight_idx = torch.tensor([discretize_birth_weight(weight) for weight in text_info['出生体重']])

    # print("gender_idx",gender_idx,"delivery_idx",delivery_idx,"asphyxia_idx",asphyxia_idx,
    #       "age_idx",age_idx,"gestational_age_idx",gestational_age_idx,"birth_weight_idx",birth_weight_idx)

    # 获取嵌入向量
    gender_emb = gender_embedding(gender_idx)
    delivery_emb = delivery_embedding(delivery_idx)
    asphyxia_emb = asphyxia_embedding(asphyxia_idx)
    age_emb = age_embedding(age_idx)
    gestational_age_emb = gestational_age_embedding(gestational_age_idx)
    birth_weight_emb = birth_weight_embedding(birth_weight_idx)

    # 拼接嵌入向量
    # text_features = torch.cat([gender_emb, delivery_emb, asphyxia_emb, age_emb, gestational_age_emb, birth_weight_emb], dim=-1)
    # text_features = torch.cat([gender_emb, delivery_emb, asphyxia_emb], dim=-1)
    text_features = torch.cat([ age_emb, gestational_age_emb, birth_weight_emb], dim=-1)
    
    return text_features


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        # model = ResNet18(num_classes=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets_Xi_Ru_txt(base_dir=args.root_path,text_file=args.text_file, split="train", num=None, transform=transforms.Compose([
        RandomGenerator_single(args.patch_size)
    ]))
    db_val = BaseDataSets_Xi_Ru_txt(base_dir=args.root_path,text_file=args.text_file, split="val")
    #yrh
    db_test = BaseDataSets_Xi_Ru_txt(base_dir=args.root_path, text_file=args.text_file,split="test")

    total_slices = len(db_train)
    labeled_slice = total_slices // 2
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    save_best_model = 'best_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            model = model.cuda()
            ema_model = ema_model.cuda()

            alpha=1.0
            data,target = sampled_batch['image'], sampled_batch['label']
            text_info=sampled_batch["text_info"]  
            # print("text_info",text_info)
 
            text_features = vectorize_text_info(sampled_batch['text_info'])
            # print("text_features.shape",text_features)
            text_features = text_features.to(device)

            data, target = data.to(device), target.to(device)

            data, target_a, target_b, lam = cutmix(data, target, alpha)
    
        
            output = model(data,text_features)


            # print("Output shape:", output.shape)
            # print("output",output)
            # print("output_feature.shape",output_feature.shape)
            # print("Text features batch.shape:", text_features_batch.shape)
            # print("text_features_batch",text_features_batch)
            # print("target,target_a, target_b",target,target_a, target_b)


            output_soft = torch.softmax(output, dim=1)  
            loss_ce = ce_loss(output, target_a) * lam + ce_loss(output, target_b) * (1. - lam)
            loss_dice = dice_loss(torch.softmax(output_soft, dim=1), target.unsqueeze(1))
            loss1 = 0.5 * (loss_ce + loss_dice)

            unlabeled_data = data[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(
                unlabeled_data) * 0.1, -0.2, 0.2)
            ema_input = unlabeled_data + noise
            with torch.no_grad():
                ema_output = ema_model(ema_input)
                ema_output_soft = torch.softmax(ema_output, dim=1) 
            
            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (output_soft[args.labeled_bs:]-ema_output_soft)**2)



            T = 8#阈值判断
            volume_batch_r = unlabeled_data.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2]).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2)            
            preds = torch.mean(preds, dim=0)  
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) 
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            threshold = 0.6
            mask = (uncertainty<threshold).float()
            uncertainty_str = str(uncertainty)
            # logging.info('uncertainty %s : threshold : %f' % (uncertainty_str, threshold))
            consistency_dist = torch.sum(mask*consistency_loss)/(2*torch.sum(mask)+1e-16)


            consistency_weight = get_current_consistency_weight(iter_num//150) 
            loss = loss1 + consistency_weight * consistency_dist
            optimizer.zero_grad()
            loss.backward(retain_graph=True)#为什么没有其他的反向传播操作仍会报错？
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))


            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                eps = 0.0001
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_classfier_Xi_Ru(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                    print("metric_list ", metric_list)
                Acc = metric_list[0] / len(db_val)
                SEN = metric_list[1] / (metric_list[1] + metric_list[2] + eps)
                PPV = metric_list[1] / (metric_list[1] + metric_list[3] + eps)
                print("Acc:{0}, SEN:{1}, PPV:{2}".format(Acc, SEN, PPV))
                logging.info("Acc:{0}, SEN:{1}, PPV:{2}".format(Acc, SEN, PPV))

                performance = metric_list[0]

    
                writer.add_scalar('info/val_mean_dice', performance, iter_num)


                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    save_best_model = save_best

                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()


            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    print("Training Finished! Strat Test!")
    ####测试代码
    #加载最优的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path_best = save_best_model
    # model_path_best = "/mnt/sdd/wjh/Fei_Bing/model/cross_pesudo_Xi_Ru_vit2/Fully_vit_num_140_labeled_Acc_7943_PPV_8090_SEN_7478/iter_9000.pth"
    model_dict = model.state_dict()
    model_test = torch.load(model_path_best, map_location = device)
    model_dict.update(model_test)
    model.load_state_dict(model_dict)
    model.eval()
    #进行测试，具体方法就是上边的val的流程
    metric_list = 0.0
    eps = 0.0001
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume_classfier_Xi_Ru(
            sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
        metric_list += np.array(metric_i)
        print("metric_list:",metric_list)
    Acc = metric_list[0] / len(db_test)
    SEN_test = metric_list[1] / (metric_list[1] + metric_list[2] + eps)
    PPV_test = metric_list[1] / (metric_list[1] + metric_list[3] + eps)

    performance_test = Acc

    writer.add_scalar('info/test_mean_dice', performance_test, iter_num)
    print('test_dice : %f' % (performance_test)) 
    logging.info('test_dice : %f' % (performance_test))
    logging.info('SEN_test : %f' % (SEN_test))
    logging.info('PPV_test: %f' % (PPV_test))
    #最后的拼接步骤
    save_path_no_grade = "../model/{}_{}_labeled".format(
        args.exp, args.labeled_num) 


    new_directory_name = save_path_no_grade + \
                "_Acc_" + str(performance_test)[2:6] + \
                "_PPV_" + str(PPV_test)[2:6] + \
                "_SEN_" + str(SEN_test)[2:6]
    counter = 0
    while os.path.exists(new_directory_name):
        counter += 1
        new_directory_name = os.path.join(new_directory_name + '_' + str(counter))
    os.rename(save_path_no_grade, new_directory_name)
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled".format(
        args.exp, args.labeled_num) 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
