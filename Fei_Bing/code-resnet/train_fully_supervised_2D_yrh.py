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
from torch.utils.data import Dataset
from PIL import Image

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
args = parser.parse_args()


# class CustomImageDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.image_names = os.listdir(image_dir)  # 获取所有图像文件名
#         self.transform = transform
#         self.target_size = target_size  # 目标尺寸

#     def __len__(self):
#         return len(self.image_names)

#     def __getitem__(self, idx):
#         image_name = self.image_names[idx]
#         image_path = os.path.join(self.image_dir, image_name)
#         mask_path = os.path.join(self.mask_dir, image_name.replace('flair', 'seg'))  # 根据规则生成掩码路径

#         # 打开图像
#         image = Image.open(image_path).convert('L')  # 图像转换为灰度
#         mask = Image.open(mask_path).convert('L')  # 掩码转换为灰度
        
#         # 对图像进行resize（使用transform中的其他变换）
#         if self.transform:
#             image = self.transform(image)

#         # 对掩码进行resize（不使用ToTensor，因为我们不希望掩码被二值化）
#         mask = mask.resize(self.target_size, Image.NEAREST)  # 使用最近邻插值调整掩码大小

#         # 将掩码转换为NumPy数组，并根据需要进行标签映射
#         mask_array = np.array(mask)
#         mask_array[mask_array == 64] = 1
#         mask_array[mask_array == 128] = 2
#         mask_array[mask_array == 255] = 3

#         # 将掩码转换为torch长整型张量
#         mask = torch.tensor(mask_array, dtype=torch.long)
#         mask = mask.unsqueeze(0)

#         return {'image': image, 'label': mask}


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = os.listdir(image_dir)  # 获取所有图像文件名
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace('flair', 'seg'))  # 根据规则生成掩码路径

        image = Image.open(image_path).convert('L')  # 打开图像
        mask = Image.open(mask_path).convert('L')  # 打开掩码，转换为单通道图像

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'label': mask}


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623,"140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    # 设置图像和掩码的路径
    image_dir_train = '/mnt/sdc/tangjiaqi/TCIA/Brain_seg/train/img/flair'
    mask_dir_train = '/mnt/sdc/tangjiaqi/TCIA/Brain_seg/train/seg'
    image_dir_val = '/mnt/sdc/tangjiaqi/TCIA/Brain_seg/val/img/flair'
    mask_dir_val = '/mnt/sdc/tangjiaqi/TCIA/Brain_seg/val/seg'
    image_dir_test = '/mnt/sdc/tangjiaqi/TCIA/Brain_seg/test/img/flair'
    mask_dir_test = '/mnt/sdc/tangjiaqi/TCIA/Brain_seg/test/seg'

    # 数据集和数据加载器
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 将图像和掩码统一大小
        transforms.ToTensor(),  # 转换为张量
    ])
    
    db_train = CustomImageDataset(image_dir=image_dir_train, mask_dir=mask_dir_train, transform=transform)
    db_val = CustomImageDataset(image_dir=image_dir_val, mask_dir=mask_dir_val, transform=transform)  # 验证集和训练集一致
    db_test = CustomImageDataset(image_dir=image_dir_test, mask_dir=mask_dir_test, transform=transform)  # 测试集也一样
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

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
            label_batch = label_batch.squeeze(1)

            # label_batch_ = label_batch.cpu().numpy()
            # label_batch_flat = label_batch_.flatten().astype(np.int32)
            # unique_labels, counts = np.unique(label_batch_flat, return_counts=True)
            # print(f"Unique labels: {unique_labels}, Counts: {counts}")
  
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

            if iter_num > 0 and iter_num % 200 == 0:
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
    # model_path_best = "/mnt/sdd/wjh/Fei_Bing/model/BraTS/seg_140_labeled_DSC_6817_PPV_7986_SEN_7754_DH95_3.0747/unet_best_model.pth"
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
        # print("metric_list:",metric_list)
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
    iou_test = np.mean(metric_list, axis=0)[4]
    boundary_iou_test = np.mean(metric_list, axis=0)[5]
    hd_test = np.mean(metric_list, axis=0)[6]
    asd_test = np.mean(metric_list,axis=0)[7]

    writer.add_scalar('info/test_mean_dice', performance_test, iter_num)
    writer.add_scalar('info/test_mean_hd95', mean_hd95_test, iter_num)
    print('test_dice : %f test_hd95 : %f' % (performance_test, mean_hd95_test)) 
    logging.info('test_dice : %f test_hd95 : %f' % (performance_test, mean_hd95_test)) 
    logging.info('PPV_test : %f SEN_test : %f miou_test : %f boundary_iou_test : %f hd_test : %f' 
                 % (PPV_test, SEN_test,iou_test,boundary_iou_test,hd_test))
    logging.info('asd_test : %f' %(asd_test))
    #最后的拼接步骤
    save_path_no_grade = "../model/{}_{}_labeled".format(
        args.exp, args.labeled_num) 
    new_directory_name = save_path_no_grade + \
        "_DSC_"+str(performance_test)[2:6]+ \
        "_PPV_"+str(PPV_test)[2:6]+ \
        "_SEN_"+str(SEN_test)[2:6]+ \
        "_DH95_"+str(mean_hd95_test)[0:6]+ \
        "_DH_"+str(hd_test)[0:6]+ \
        "_iou_"+str(iou_test)[2:6]+ \
        "_boundary_iou_"+str(boundary_iou_test)[:6]
    counter = 1
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