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
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from networks.vitttt import ViT

from dataloaders import utils
from dataloaders.dataset_yrh import BaseDataSets, RandomGenerator, BaseDataSets_xueyoubing, RandomGenerator_single, BaseDataSets_Xi_Ru
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_volume_classfier, test_single_volume_classfier_Xi_Ru
from cutmix.cutmix import cutmix
from networks.resnet import ResNet18,ResNet34,ResNet18
from networks.convnext import ConvNeXt,convnext_tiny
from networks.vgg import vgg16,VGG16
from networks.xception import Xception
from networks.EfficientNet import efficientnet_b0
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='DenseNet_2_class', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,# 吸入综合征的是二分类
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.003,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=14, help='random seed')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')
parser.add_argument('--alg', type=str, default='supervised', help='Algorithm type')
parser.add_argument('--reweight', type=str, default='None', help='Reweighting strategy for class imbalance')
parser.add_argument('--sntg', type=float, default=0, help='SNTG coefficient')
parser.add_argument("--cb-beta", default=0.99, type=float, help="hyperparameter of class-balanced loss (default: 0.9999).")
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

def reset_stats(device):
    torch.cuda.reset_peak_memory_stats(device)
    return time.time()

def log_stats(start_time, device, tag='batch'):
    elapsed = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
    logging.info(f'[{tag}] time: {elapsed:.3f}s  |  peak GPU mem: {peak_mem:.2f} GB')


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    # model = ResNet18(num_classes=num_classes)
    # model = convnext_tiny(pretrained=True, in_22k=False, num_classes=2, in_chans=1)
    # model = vgg16(num_classes=num_classes) 
    # model = Xception(num_classes=num_classes)
    # model = efficientnet_b0(num_classes=num_classes)
    # model = ViT(num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cpu().eval()
    # 计算理论复杂度
    input_res = (1, args.patch_size[0], args.patch_size[1])
    macs, params = get_model_complexity_info(
        model, input_res,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
    logging.info(f'Complexity  Params: {params}  |  FLOPs: {macs}')    
    model = model.to(device).train()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    db_train = BaseDataSets_Xi_Ru(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator_single(args.patch_size)
    ]))
    db_val = BaseDataSets_Xi_Ru(base_dir=args.root_path, split="val")
    db_test = BaseDataSets_Xi_Ru(base_dir=args.root_path, split="test")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
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
    model.to(device)

    for epoch_num in iterator:
        epoch_start = reset_stats(device)        
        for i_batch, sampled_batch in enumerate(trainloader):
            batch_start = reset_stats(device)

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            print("volume_batch.shape",volume_batch.shape)

            # alpha=1.0
            # data,target = sampled_batch['image'], sampled_batch['label']
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # data, target = data.to(device), target.to(device)
            # criteria = nn.CrossEntropyLoss()
            # data, target_a, target_b, lam = cutmix(data, target, alpha)
            # output = model(data)
            # loss = criteria(output, target_a) * lam + criteria(output, target_b) * (1. - lam)

            outputs = model(volume_batch)
            target = label_batch.long()
            
            # condition = {}
            # if args.cb_beta > 0:
            #    condition["cb-beta"] = args.cb_beta

            # n_labels_per_class = torch.bincount(target[target != -1])
            # n_labels_per_class = torch.Tensor(n_labels_per_class).cuda()
            # cb_weight = (1 - args.cb_beta) / (1 - args.cb_beta ** n_labels_per_class)
            # n_labels_per_class = n_labels_per_class.float()
            # # print("n_labels_per_class",n_labels_per_class)
            # # print("cb_weight",cb_weight)
            # total_labels_per_class = torch.Tensor([475, 392]).cuda()


            # if args.reweight == "inverse":
            #     cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
            # elif args.reweight == "None":
            #     inverse_weight = torch.sum(total_labels_per_class) / total_labels_per_class
            #     re_inverse_weight = inverse_weight * len(total_labels_per_class) / sum(inverse_weight)
            #     target_weights = torch.stack([re_inverse_weight[t] for t in target])
            #     cls_loss = (target_weights * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()
            # elif args.reweight == "cls_bal":
            #     target_weights = torch.stack([cb_weight[t] for t in target])
            #     cls_loss = (target_weights * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()
            # elif args.reweight == "focal":
            #     softmax_value = F.softmax(outputs, dim=-1)
            #     pt = softmax_value.gather(1, target.view(-1, 1)).squeeze()
            #     cls_loss = (((1 - pt) ** 1) * F.cross_entropy(outputs, target, reduction="none", ignore_index=-1)).mean()

            # loss = cls_loss



            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch == 0:
                #记录单次batch情况
                log_stats(batch_start, device, tag=f'epoch{epoch_num}-batch{i_batch}')

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            # writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            # if iter_num % 20 == 0:
            #     image = volume_batch[1, 0:1, :, :]
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(
            #         outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction',
            #                      outputs[1, ...] * 50, iter_num)
            #     labs = label_batch[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        #     if iter_num > 0 and iter_num % 200 == 0:
        #         model.eval()
        #         metric_list = 0.0
        #         eps = 0.0001
        #         for i_batch, sampled_batch in enumerate(valloader):
        #             metric_i = test_single_volume_classfier_Xi_Ru(
        #                 sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
        #             metric_list += np.array(metric_i)
        #             print("metric_list ", metric_list)
        #         #tp（True Positives）正确预测为正类的样本 fn（ False Negatives）实际为正类但被错误预测为负类 fp（False Positives）际为负类但被错误预测为正类
        #         Acc = metric_list[0] / len(db_val)#准确率 
        #         SEN = metric_list[1] / (metric_list[1] + metric_list[2] + eps)#召回率 = tp /（tp + fn）
        #         PPV = metric_list[1] / (metric_list[1] + metric_list[3] + eps)#回归 = tp / （tp +fp ）
        #         print("Acc:{0}, SEN:{1}, PPV:{2}".format(Acc, SEN, PPV))
        #         logging.info("Acc:{0}, SEN:{1}, PPV:{2}".format(Acc, SEN, PPV))
        #         # logging.info("Acc : %s", Acc)

        #         # for class_i in range(num_classes-1):
        #         #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
        #         #                       metric_list[class_i, 0], iter_num)
        #         #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
        #         #                       metric_list[class_i, 1], iter_num)
        #         #     writer.add_scalar('info/val_{}_ppv'.format(class_i+1),
        #         #                       metric_list[class_i, 2], iter_num)
        #         #     writer.add_scalar('info/val_{}_sen'.format(class_i+1),
        #         #                       metric_list[class_i, 3], iter_num)

        #         # performance = np.mean(metric_list, axis=0)[0]
        #         performance = metric_list[0]

        #         # mean_hd95 = np.mean(metric_list, axis=0)[1]
        #         writer.add_scalar('info/val_mean_dice', performance, iter_num)
        #         # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

        #         if performance > best_performance:
        #             best_performance = performance
        #             save_mode_path = os.path.join(snapshot_path,
        #                                           'iter_{}_dice_{}.pth'.format(
        #                                               iter_num, round(best_performance, 4)))
        #             save_best = os.path.join(snapshot_path,
        #                                      '{}_best_model.pth'.format(args.model))
        #             torch.save(model.state_dict(), save_mode_path)
        #             torch.save(model.state_dict(), save_best)
        #             save_best_model = save_best

        #         logging.info(
        #             'iteration %d : mean_dice : %f' % (iter_num, performance))
        #         model.train()

        #     if iter_num % 3000 == 0:
        #         save_mode_path = os.path.join(
        #             snapshot_path, 'iter_' + str(iter_num) + '.pth')
        #         torch.save(model.state_dict(), save_mode_path)
        #         logging.info("save model to {}".format(save_mode_path))

        #     if iter_num >= max_iterations:
        #         break
        # if iter_num >= max_iterations:
        #     iterator.close()
        #     break
        log_stats(epoch_start, device, tag=f'epoch{epoch_num}-total')            
    print("Training Finished! Strat Test!")
    ####测试代码
    #加载最优的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path_best = save_best_model
    # model_path_best = "/mnt/sdd/wjh/Fei_Bing/model/experiment/Fully_DenseNet50_Xi_Ru_140_labeled/DenseNet_2_class_best_model.pth"
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
    logging.info('PPV_test : %f' % (PPV_test))
    # #最后的拼接步骤
    save_path_no_grade = "../model/{}_{}_labeled".format(
        args.exp, args.labeled_num) 
    new_directory_name = save_path_no_grade + \
                "_Acc_" + str(performance_test)[2:6] + \
                "_PPV_" + str(SEN_test)[2:6] + \
                "_SEN_" + str(PPV_test)[2:6]
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
