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
from dataloaders.dataset_yrh import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
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
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')

parser.add_argument('--break_layer', type=str, default = 'encoder.down2.maxpool_conv.1.conv_conv.0.weight',
                    help='break_layer')
parser.add_argument('--break_ratio', type=float, default=0.0,
                    help='break_ratio')
parser.add_argument('--break_iter_num', type=int, default=200,
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


def ema_param1(model, break_layer, break_ratio):
    
    param_to_encoder = None
    param_to_decoder = None
    for name, param in model.named_parameters():
        if name == 'encoder.down1.maxpool_conv.1.conv_conv.0.weight':
            param_to_decoder = param.reshape(16, 32, 3, 3)
        if name == 'decoder.up4.conv.conv_conv.0.weight':
            param_to_encoder = param.reshape(32, 16, 3, 3)
    for name, param in model.named_parameters():
        if name == 'encoder.down1.maxpool_conv.1.conv_conv.0.weight':
            param.data = 0.99 * param.data + 0.01 * param_to_encoder
        if name == 'decoder.up4.conv.conv_conv.0.weight':
            param.data = 0.99 * param.data + 0.01 * param_to_decoder
            
def ema_param2(model, break_layer, break_ratio):
    
    param_to_encoder = None
    param_to_decoder = None
    
    for name, param in model.named_parameters():
        if name == 'encoder.down2.maxpool_conv.1.conv_conv.0.weight':
            #print('name{}, param_ori{}', name, param.data)
            param_to_decoder = param.reshape(32, 64, 3, 3)
            #print('name{}, param_broken{}',name, param.data)
        if name == 'decoder.up3.conv.conv_conv.0.weight':
            param_to_encoder = param.reshape(64, 32, 3, 3)
    for name, param in model.named_parameters():
        if name == 'encoder.down2.maxpool_conv.1.conv_conv.0.weight':
            param.data = 0.9 * param.data + 0.1 * param_to_encoder
        if name == 'decoder.up3.conv.conv_conv.0.weight':
            param.data = 0.9 * param.data + 0.1 * param_to_decoder
def ema_param3(model, break_layer, break_ratio):
    
    param_to_encoder = None
    param_to_decoder = None
    for name, param in model.named_parameters():
        if name == 'encoder.down3.maxpool_conv.1.conv_conv.0.weight':
            param_to_decoder = param.reshape(64, 128, 3, 3)
        if name == 'decoder.up2.conv.conv_conv.0.weight':
            param_to_encoder = param.reshape(128, 64, 3, 3)
    for name, param in model.named_parameters():
        if name == 'encoder.down3.maxpool_conv.1.conv_conv.0.weight':
            param.data = 0.9 * param.data + 0.1 * param_to_encoder
        if name == 'decoder.up2.conv.conv_conv.0.weight':
            param.data = 0.9 * param.data + 0.1 * param_to_decoder
def ema_param4(model, break_layer, break_ratio):
    
    param_to_encoder = None
    param_to_decoder = None
    for name, param in model.named_parameters():
        if name == 'encoder.down4.maxpool_conv.1.conv_conv.0.weight':
            #print('name{}, param_ori{}', name, param.data)
            param_to_decoder = param.reshape(128, 256, 3, 3)
            #print('name{}, param_broken{}',name, param.data)
            #print('have broken the layer{}!!!',name)
        if name == 'decoder.up1.conv.conv_conv.0.weight':
            param_to_encoder = param.reshape(256, 128, 3, 3)
    for name, param in model.named_parameters():
        if name == 'encoder.down4.maxpool_conv.1.conv_conv.0.weight':
            param.data = 0.9 * param.data + 0.1 * param_to_encoder
        if name == 'decoder.up1.conv.conv_conv.0.weight':
            param.data = 0.9 * param.data + 0.1 * param_to_decoder

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=labeled_slice, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    #yrh
    db_test = BaseDataSets(base_dir=args.root_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    
    model.train()

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
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if iter_num == args.break_iter_num or iter_num == args.break_iter_num + 2000 or iter_num == args.break_iter_num +4000:
            if iter_num >= args.break_iter_num and (iter_num % 20 == 5 or iter_num % 20 == 19):
                ema_param4(model, args.break_layer, args.break_ratio)
                print('ema 4!!!!!!!!')
            if iter_num >= args.break_iter_num and (iter_num % 20 == 4 or iter_num % 20 == 10):
                ema_param3(model, args.break_layer, args.break_ratio)
                print('ema 3!!!!!!!!')
            if iter_num >= args.break_iter_num and (iter_num % 20 == 9 or iter_num % 20 == 15):
                ema_param2(model, args.break_layer, args.break_ratio)
                print('ema 2!!!!!!!!')
            if iter_num >= args.break_iter_num and (iter_num % 20 == 14 or iter_num % 20 == 0):
                ema_param1(model, args.break_layer, args.break_ratio)
                print('ema 1!!!!!!!!')
                
            
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            #if iter_num > 0 and iter_num % 200 == 0:
            if (iter_num > 0 and iter_num % 200 == 0) or (iter_num >= args.break_iter_num and iter_num <= args.break_iter_num + 300 and iter_num % 50 == 0):
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_ppv'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_sen'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                #yrh
                if iter_num >= args.break_iter_num and iter_num <args.break_iter_num + 300 and iter_num % 20 == 0:
                    save_mode_path = os.path.join(snapshot_path,
                                                  'break_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(performance, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                
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
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
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
    model_dict = model.state_dict()
    model_test = torch.load(model_path_best, map_location = device)
    model_dict.update(model_test)
    model.load_state_dict(model_dict)
    model.eval()
    #进行测试，具体方法就是上边的val的流程
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume(
            sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
        metric_list += np.array(metric_i)
        print("metric_list:",metric_list)
    metric_list = metric_list / len(db_test)
    for class_i in range(num_classes-1):
        writer.add_scalar('info/test_{}_dice'.format(class_i+1),
                          metric_list[class_i, 0], iter_num)
        writer.add_scalar('info/test_{}_hd95'.format(class_i+1),
                          metric_list[class_i, 1], iter_num)

    performance_test = np.mean(metric_list, axis=0)[0]
    mean_hd95_test = np.mean(metric_list, axis=0)[1]
    PPV_test = np.mean(metric_list, axis=0)[2]
    SEN_test = np.mean(metric_list, axis=0)[3]

    writer.add_scalar('info/test_mean_dice', performance_test, iter_num)
    writer.add_scalar('info/test_mean_hd95', mean_hd95_test, iter_num)
    print('test_dice : %f test_hd95 : %f' % (performance_test, mean_hd95_test)) 
    logging.info('test_dice : %f test_hd95 : %f' % (performance_test, mean_hd95_test)) 
    #最后的拼接步骤
    save_path_no_grade = "../model/{}_{}_labeled".format(
        args.exp, args.labeled_num) 
    os.rename(save_path_no_grade,save_path_no_grade + "_DSC_"+str(performance_test)[2:6]+"_PPV_"+str(PPV_test)[2:6]+"_SEN_"+str(SEN_test)[2:6]+"_DH95_"+str(mean_hd95_test)[0:6]) 
    
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
