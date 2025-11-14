import argparse
import logging
import os
import random
import shutil
import sys
import time

from matplotlib import pyplot as plt
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
from val_2D import test_single_volume, test_single_volume_ds, test_single_volume_classfier, test_single_volume_classfier_Xi_Ru,test_single_volume_classfier_Fei_Tou
from dataloaders.dataset_yrh import BaseDataSets, RandomGenerator, BaseDataSets_xueyoubing, RandomGenerator_single, BaseDataSets_Xi_Ru, BaseDataSets_Fei_Tou
from cutmix.cutmix import cutmix
from PIL import Image 

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Mean_Teacher', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='DenseNet_5_class_Fei_Tou', help='model_name')
parser.add_argument('--num_classes', type=int,  default=5, # 肺透明膜病的分类是5，直接用Xi_Ru 的网络，只修改分类数就行
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
parser.add_argument('--labeled_bs', type=int, default=2,
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
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


# 定义绘制趋势图的函数
def plot_uncertainty_trend(uncertainty_list, save_path="./uncertainty_trend_Fei_Tou.png"):
    plt.figure(figsize=(10, 6))
    
    # 获取图像数量和epoch数量
    num_images = uncertainty_list[0].shape[0]  # 每个epoch有8张图像
    epochs = range(len(uncertainty_list))  # epoch的数量

    # 对每张图像的不确定度趋势绘制一条曲线
    for img_idx in range(num_images):
        # 提取每个epoch中该图像的不确定度值
        img_uncertainty = [uncertainty[img_idx].item() for uncertainty in uncertainty_list]
        plt.plot(epochs, img_uncertainty, label=f"Image {img_idx + 1}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty Trend Over Epochs for Each Image')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 保存图像
    plt.close()


# 将张量转换为 NumPy 数组并保存图像
def save_cutmix_image(tensor, epoch, batch, save_path="./cutmix_images_Fei_Tou"):
    # 将张量从 GPU 转到 CPU，并转换为 NumPy 格式
    tensor = tensor.cpu().detach()
    
    # # 将张量的通道从 [C, H, W] 转换为 [H, W, C]
    # grid_image = make_grid(tensor, nrow=4).permute(1, 2, 0).numpy()#make_grid将一个batch的处理成一张大的拼接图像

    # # 反归一化（假设数据经过了标准化处理）
    # grid_image = grid_image * 255.0
    # grid_image = grid_image.astype(np.uint8)

    # 遍历 batch 中的每一张图像
    for idx, img_tensor in enumerate(tensor):
        # 将张量的通道从 [C, H, W] 转换为 [H, W, C]
        img_numpy = img_tensor.permute(1, 2, 0).numpy()

        # 反归一化（假设数据经过了标准化处理）
        img_numpy = img_numpy * 255.0
        img_numpy = img_numpy.astype(np.uint8)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 使用 matplotlib 保存每张图像
        plt.figure(figsize=(5, 5))  # 每张图像大小
        plt.imshow(img_numpy)
        plt.axis("off")
        plt.title(f"CutMix Epoch {epoch} Batch {batch} Image {idx}")
        plt.savefig(f"{save_path}/cutmix_epoch_{epoch}_batch_{batch}_image_{idx}.png")
        plt.close()


def load_images(image_dir, epoch, batch, required_images=8):
    # 准备存储图像的列表
    image_list = []
    
    # 定义转换操作，将图像转换为 PyTorch 张量并进行归一化
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图像
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图像归一化
    ])

    # 定义初始的批次和图像索引
    current_batch = batch
    image_idx = 0

    # 循环直到我们获得所需数量的图像
    while len(image_list) < required_images:
        image_path = os.path.join(image_dir, f"cutmix_epoch_{epoch}_batch_{current_batch}_image_{image_idx}.png")
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB
            img_tensor = transform(img)  # 转换为张量并归一化
            image_list.append(img_tensor)
            image_idx += 1  # 读取下一个图像
        else:
            # 当前批次的图像已读取完，切换到下一个批次
            current_batch += 1
            image_idx = 0  # 重置图像索引
    
    # 将图像列表转换为一个批次的张量 [batch_size, channels, height, width]
    images_tensor = torch.stack(image_list[:required_images], dim=0)  # 只保留所需的数量

    return images_tensor


def plot_accuracy_bar_chart(accuracy_list):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(accuracy_list) + 1)
    
    plt.bar(epochs, accuracy_list, color='skyblue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend Over Epochs')
    
    # 显示每个柱上的准确率数值
    for i, acc in enumerate(accuracy_list):
        plt.text(i + 1, acc + 0.01, f'{acc:.2f}', ha='center')
    
    plt.grid(True)
    plt.savefig("accuracy_trend.png")  # 保存图像
    plt.show()


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

    db_train = BaseDataSets_Fei_Tou(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator_single(args.patch_size)
    ]))
    db_val = BaseDataSets_Fei_Tou(base_dir=args.root_path, split="val")
    #yrh
    db_test = BaseDataSets_Fei_Tou(base_dir=args.root_path, split="test")

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
    
    uncertainty_list = []  # 用于存储不确定度值
    threshold_list = []
    accuracy_list = []

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
    
            model = model.cuda()
            ema_model = ema_model.cuda()

            alpha=1.0
            data,target = sampled_batch['image'], sampled_batch['label']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data, target = data.to(device), target.to(device)
            criteria = nn.CrossEntropyLoss()

            non_zero_indices = torch.nonzero(target).squeeze(dim=1)  # 获取非零索引
            non_zero_data = data[non_zero_indices]
            non_zero_target = target[non_zero_indices]
            zero_indices = torch.nonzero(target == 0).squeeze(dim=1)  # 获取零索引
            zero_data = data[zero_indices]
            zero_target = target[zero_indices]
            if zero_target.numel() > 0 and non_zero_target.numel() > 0:
                data_mix, target_a_mix, target_b_mix, lam_mix = cutmix(zero_data, zero_target, alpha)
                mixed_data = torch.cat((data_mix, non_zero_data), dim=0)
                mixed_target_a = torch.cat((target_a_mix, non_zero_target), dim=0)
                mixed_target_b = torch.cat((target_b_mix, non_zero_target), dim=0)

            #获取CutMix操作后的图像
            # save_cutmix_image(mixed_data, epoch_num, i_batch)

            output = model(mixed_data)
            output_soft = torch.softmax(output, dim=1)                
            loss_ce = criteria(output, mixed_target_a) * lam_mix + criteria(output, mixed_target_b) * (1. - lam_mix)
            loss_dice = dice_loss(output_soft[:], target[:].unsqueeze(1))             
            loss1 = 0.5*(loss_ce + loss_dice)


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



            # T = 8#阈值判断
            # volume_batch_r = unlabeled_data.repeat(2, 1, 1, 1)

            # stride = volume_batch_r.shape[0] // 2

            # preds = torch.zeros([stride * T, 5]).cuda()

            # for i in range(T//2):
            #     ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)

            #     with torch.no_grad():
            #         preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            # preds = F.softmax(preds, dim=1)
            # preds = preds.reshape(T, stride, 5)
            # preds = torch.mean(preds, dim=0)  
            # uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) 

            # threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(5)
            # # threshold = 1.5
            # mask = (uncertainty<threshold).float()

            # consistency_dist = torch.sum(mask*consistency_loss)/(2*torch.sum(mask)+1e-16)


            consistency_weight = get_current_consistency_weight(iter_num//150)
            # loss = loss1 + consistency_weight * consistency_dist 
            loss = loss1 + consistency_weight * consistency_loss           

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)



            #开始对输入图像进行可视化输出不确定度和概率
            if i_batch == 0:
                image_dir = "/mnt/sdd/wjh/Fei_Bing/code-resnet/cutmix_images_Fei_Tou"
                epoch = 0
                batch = 5

                # 加载图像
                input_images = load_images(image_dir, epoch, batch).cuda()

                with torch.no_grad():
                    output = model(input_images)  # 推理得到输出
                    output_soft = torch.softmax(output, dim=1)  # 概率分布
                T = 8  # 多次推理以计算不确定性
                volume_batch_r = input_images.repeat(2, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, 5]).cuda()  # 假设有2个类别
                for i in range(T // 2):
                    noisy_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[ 2*stride * i: 2*stride * (i + 1)] = ema_model(noisy_inputs)

                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 5)
                preds = torch.mean(preds, dim=0)  # 计算T次推理的平均值
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # 不确定性
                uncertainty_list.append(uncertainty.cpu().numpy())  # 保存不确定度数据
                plot_uncertainty_trend(uncertainty_list)
                # logging.info(f"Uncertainty_list: {uncertainty_list}")
                threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(5)
                threshold_list.append(threshold)
                logging.info(f"threshold_list:{threshold_list}")


                # 4. 日志记录
                for i in range(input_images.shape[0]):
                    logging.info(f"Image {i+1}:")
                    logging.info(f"Output probabilities: {output_soft[i]}")
                    logging.info(f"Uncertainty: {uncertainty[i]}")


                # 定义批次大小
                batch_size = 2  # 可以根据你的GPU情况调整大小
                num_samples = 100

                # 记录随机100个样本的标注结果与预测结果
                sample_indices = torch.randperm(len(trainloader.dataset))[:num_samples]
                selected_data = [trainloader.dataset[i] for i in sample_indices]  # 使用固定索引获取样本
                manual_labels = [trainloader.dataset[i]['label'] for i in sample_indices]

                # 分批处理
                selected_output_labels = []
                for i in range(0, num_samples, batch_size):
                    batch_images = torch.stack([trainloader.dataset[j]['image'] for j in sample_indices[i:i + batch_size]]).to(device)
                    output = model(batch_images)
                    batch_output_labels = torch.argmax(output, dim=1).cpu().numpy()
                    selected_output_labels.extend(batch_output_labels)

                # 计算准确率
                correct = (np.array(selected_output_labels) == np.array(manual_labels)).sum()
                accuracy = correct / num_samples
                # logging.info(f"Epoch {epoch_num}, Accuracy on 100 samples: {accuracy:.2f}")
                accuracy_list.append(accuracy)
                logging.info(f"accuracy_list:{accuracy_list}")
                plot_accuracy_bar_chart(accuracy_list)





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
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))


            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                eps = 0.0001
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_classfier_Fei_Tou(
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
    # model_path_best = "/mnt/sdd/wjh/Fei_Bing/model/cross_pesudo_fei_tou2/Fully_DenseNet50_num_140_labeled_Acc_.697_1/DenseNet_5_class_Fei_Tou_best_model.pth"
    model_dict = model.state_dict()
    model_test = torch.load(model_path_best, map_location = device)
    model_dict.update(model_test)
    model.load_state_dict(model_dict)
    model.eval()
    #进行测试，具体方法就是上边的val的流程
    metric_list = 0.0
    eps = 0.0001
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume_classfier_Fei_Tou(
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
