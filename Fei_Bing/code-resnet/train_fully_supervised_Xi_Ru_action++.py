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

from dataloaders import utils
from dataloaders.dataset_yrh import BaseDataSets, RandomGenerator, BaseDataSets_xueyoubing, RandomGenerator_single, BaseDataSets_Xi_Ru
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds, test_single_volume_classfier, test_single_volume_classfier_Xi_Ru
from cutmix.cutmix import cutmix
from networks.resnet import ResNet18,ResNet34
from networks.convnext import ConvNeXt,convnext_tiny
from networks.vgg import vgg16
from networks.xception import Xception
from networks.EfficientNet import efficientnet_b0
from dataloaders.dataset_yrh import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)

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
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
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


def compute_optimal_centers(num_classes, feature_dim):
    # 假设使用一些离线算法来计算均匀分布的类中心
    # 这里只是一个简单的示例，实际可能需要更复杂的计算
    centers = np.random.randn(num_classes, feature_dim)
    centers = torch.tensor(centers).float()
    centers = F.normalize(centers, dim=1)
    return centers


class AdaptiveSupervisedContrastiveLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, temperature=0.1, base_temperature=0.07, momentum=0.5):
        super(AdaptiveSupervisedContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.momentum = momentum
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.register_buffer('centers_momentum', torch.zeros(num_classes, feature_dim))

    def forward(self, features, labels, step):
        # Ensure all tensors are on the same device
        device = features.device
        self.centers.data = self.centers.data.to(device)
        self.centers_momentum = self.centers_momentum.to(device)

        # Normalize the features
        features = F.normalize(features, dim=1)
        
        # Calculate class centers (prototypes)
        centers = torch.zeros(self.num_classes, self.feature_dim).to(device)
        for c in range(self.num_classes):
            class_features = features[labels == c]
            if len(class_features) > 0:
                centers[c] = class_features.mean(dim=0)
        
        # Normalize the class centers
        centers = F.normalize(centers, dim=1)
        
        # Compute pairwise similarity between features and centers
        similarity_matrix = torch.matmul(features, centers.T)

        # Adaptive temperature using cosine scheduler
        dynamic_temperature = self._cosine_scheduler(step).to(device)
        dynamic_temperature = dynamic_temperature.unsqueeze(1)  # Adjust shape to (num_classes, 1)

        # Print the shapes of tensors for debugging
        print(f"similarity_matrix shape: {similarity_matrix.shape}")
        print(f"dynamic_temperature shape: {dynamic_temperature.shape}")

        # Hard queries sampling: select features farthest from the class center
        hard_queries = []
        for c in range(self.num_classes):
            class_features = features[labels == c]
            if len(class_features) > 0:
                distances = torch.norm(class_features - centers[c], dim=1)
                hard_query = class_features[distances.argmax()]
                hard_queries.append(hard_query)
        hard_queries = torch.stack(hard_queries)
        
        # Negative sampling: select negative samples from other classes
        neg_samples = []
        for i in range(self.num_classes):
            neg_indices = labels != i
            neg_features = features[neg_indices]
            neg_samples.append(neg_features)
        
        loss = 0
        for i, hard_query in enumerate(hard_queries):
            pos_sim = F.cosine_similarity(hard_query.unsqueeze(0), centers[i].unsqueeze(0))
            neg_sim = torch.cat([F.cosine_similarity(hard_query.unsqueeze(0), neg) for neg in neg_samples if len(neg) > 0])
            
            # Compute cross-entropy loss
            logits = torch.cat([pos_sim, neg_sim])
            logits = logits / dynamic_temperature.expand_as(logits)  # Broadcast dynamic_temperature to match logits
            labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)
            labels[0] = 1  # The first logit is the positive example
            loss += F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0), reduction='sum')
        
        # Update centers with momentum
        for i in range(len(labels)):
            self.centers_momentum[labels[i]] = self.momentum * self.centers_momentum[labels[i]] + (1 - self.momentum) * features[i]
            self.centers.data[labels[i]] = self.centers_momentum[labels[i]]

        return loss / len(hard_queries)

    def _cosine_scheduler(self, step):
        # Simple cosine scheduler for dynamic temperature τ
        return self.base_temperature + 0.5 * (self.temperature - self.base_temperature) * (1 + torch.cos(torch.tensor(step * 3.14159265359 / 100, device=self.centers.device)))


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)




def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

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

    # model = ResNet18(num_classes=num_classes)
    # model = convnext_tiny(pretrained=True, in_22k=False, num_classes=2, in_chans=1)
    # model = vgg16(num_classes=num_classes) 
    # model = Xception(num_classes=num_classes)
    # model = efficientnet_b0(num_classes=num_classes)

    db_train = BaseDataSets_Xi_Ru(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator_single(args.patch_size)
    ]))
    db_val = BaseDataSets_Xi_Ru(base_dir=args.root_path, split="val")
    #yrh
    db_test = BaseDataSets_Xi_Ru(base_dir=args.root_path, split="test")
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)


    total_slices = len(db_train)
    labeled_slice = total_slices // 2
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)


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
            # print("volume_batch.shape",volume_batch.shape)

            # alpha=1.0
            # data,target = sampled_batch['image'], sampled_batch['label']
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # data, target = data.to(device), target.to(device)
            # criteria = nn.CrossEntropyLoss()
            # data, target_a, target_b, lam = cutmix(data, target, alpha)
            # output = model(data)
            # loss = criteria(output, target_a) * lam + criteria(output, target_b) * (1. - lam)

            model = model.cuda()
            ema_model = ema_model.cuda()

            features,outputs = model(volume_batch) 
            feature_dim = model.classifier_1.in_features
            optimal_centers = compute_optimal_centers(num_classes, feature_dim).cuda()
            contrastive_loss_fn = AdaptiveSupervisedContrastiveLoss(num_classes, feature_dim,optimal_centers)


            target = label_batch.long()


            output_soft = torch.softmax(outputs, dim=1) 
            criteria = nn.CrossEntropyLoss()
            loss_ce = criteria(outputs[:], label_batch[:][:].long())
            loss_dice = dice_loss(output_soft[:], target[:].unsqueeze(1))             
            loss1 = 0.5*(loss_ce + loss_dice)

            unlabeled_data = volume_batch[args.labeled_bs:]
            noise = torch.clamp(torch.randn_like(
                unlabeled_data) * 0.1, -0.2, 0.2)
            ema_input = unlabeled_data + noise
            with torch.no_grad():
                ema_features,ema_output = ema_model(ema_input)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (output_soft[args.labeled_bs:]-ema_output_soft)**2)


            consistency_weight = get_current_consistency_weight(iter_num//150)
            step = 10

            loss = contrastive_loss_fn(features, target, step)*0.5 + loss1 + consistency_weight * consistency_loss


            # criteria = nn.CrossEntropyLoss()
            # loss_1 = criteria(outputs[:], label_batch[:][:].long())
            # loss = loss_1
            # print("outputs", outputs)
            # 
            # print("label_batch type ", label_batch)
            # outputs_soft = torch.softmax(outputs, dim=1)
            # print("outputs[0]", outputs[0])
            # print("label_batch[0]", label_batch[0])
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            # loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            # param_group['lr'] = lr_

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

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                eps = 0.0001
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_classfier_Xi_Ru(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                    print("metric_list ", metric_list)
                #tp（True Positives）正确预测为正类的样本 fn（ False Negatives）实际为正类但被错误预测为负类 fp（False Positives）际为负类但被错误预测为正类
                Acc = metric_list[0] / len(db_val)#准确率 
                SEN = metric_list[1] / (metric_list[1] + metric_list[2] + eps)#召回率 = tp /（tp + fn）
                PPV = metric_list[1] / (metric_list[1] + metric_list[3] + eps)#回归 = tp / （tp +fp ）
                print("Acc:{0}, SEN:{1}, PPV:{2}".format(Acc, SEN, PPV))
                logging.info("Acc:{0}, SEN:{1}, PPV:{2}".format(Acc, SEN, PPV))
                # logging.info("Acc : %s", Acc)

                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                #                       metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                #                       metric_list[class_i, 1], iter_num)
                #     writer.add_scalar('info/val_{}_ppv'.format(class_i+1),
                #                       metric_list[class_i, 2], iter_num)
                #     writer.add_scalar('info/val_{}_sen'.format(class_i+1),
                #                       metric_list[class_i, 3], iter_num)

                # performance = np.mean(metric_list, axis=0)[0]
                performance = metric_list[0]

                # mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                # writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

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
