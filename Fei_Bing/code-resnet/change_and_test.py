import argparse
import logging
import os
import random
import shutil
import sys
import time

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
from val_2D import test_single_volume, test_single_volume_change_skip_concat

import random

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_change_skip_concat', help='model_name')
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
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
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
#yrh_add
parser.add_argument('--break_layer', type=str, default = 'encoder.down2.maxpool_conv.1.conv_conv.0.weight',
                    help='break_layer')
parser.add_argument('--break_ratio', type=float, default=0.0,
                    help='break_ratio')
parser.add_argument('--break_iter_num', type=int, default=1000,
                    help='break_iter_num')

args = parser.parse_args()



def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    """ #查看名字和参数
    for name, param in model.named_parameters():
        print(name, param.shape) """
        
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def break_param3(model, ema_model, break_layer, break_ratio):
    
    param_to_encoder = None
    param_to_decoder = None
    for name, param in model.named_parameters():
        if name == 'encoder.down3.maxpool_conv.1.conv_conv.0.weight':
            param_to_decoder = param.reshape(64, 128, 3, 3)
        if name == 'decoder.up2.conv.conv_conv.0.weight':
            param_to_encoder = param.reshape(128, 64, 3, 3)
    for name, param in model.named_parameters():
        if name == 'encoder.down3.maxpool_conv.1.conv_conv.0.weight':
            param.data = param_to_encoder
        if name == 'decoder.up2.conv.conv_conv.0.weight':
            param.data = param_to_decoder 

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    db_test = BaseDataSets(base_dir=args.root_path, split="test")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)

    iter_num = 0
    
    save_best_model = '/mnt/sdd/yrh/code_star/0313_change_and_test_ACDC/pth/Cocoon_num1_lr0.9_0.1_7_labeled_DSC_8848_PPV_8814_SEN_8957_DH95_4.3986_DH_21.178_iou_7999_boundary_iou_0.4712/iter_14200_dice_0.8716.pth'
    skip_concat = True
    
    ####测试代码
    #加载最优的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path_best = save_best_model
    model_dict = model.state_dict()
    model_test = torch.load(model_path_best, map_location = device)
    model_dict.update(model_test)
    model.load_state_dict(model_dict)
    model.eval()
    #进行测试，具体方法就是上边的val的流程
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume_change_skip_concat(
            sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
        metric_list += np.array(metric_i)
        print("metric_list:",metric_list)
    metric_list = metric_list / len(db_test)

    performance_test = np.mean(metric_list, axis=0)[0]
    mean_hd95_test = np.mean(metric_list, axis=0)[1]
    PPV_test = np.mean(metric_list, axis=0)[2]
    SEN_test = np.mean(metric_list, axis=0)[3]
    iou_test = np.mean(metric_list, axis=0)[4]
    boundary_iou_test = np.mean(metric_list, axis=0)[5]
    hd_test = np.mean(metric_list, axis=0)[6]

    print('test_dice : %f test_hd95 : %f' % (performance_test, mean_hd95_test)) 
    logging.info('test_dice : %f test_hd95 : %f' % (performance_test, mean_hd95_test)) 
    #最后的拼接步骤
    save_path_no_grade = "../model/{}_{}_labeled".format(
        args.exp, args.labeled_num) 
    os.rename(save_path_no_grade,save_path_no_grade + \
        "_DSC_"+str(performance_test)[2:6]+ \
        "_PPV_"+str(PPV_test)[2:6]+ \
        "_SEN_"+str(SEN_test)[2:6]+ \
        "_DH95_"+str(mean_hd95_test)[0:6]+ \
        "_DH_"+str(hd_test)[0:6]+ \
        "_iou_"+str(iou_test)[2:6]+ \
        "_boundary_iou_"+str(boundary_iou_test)[:6])
    return "Finished!"


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
