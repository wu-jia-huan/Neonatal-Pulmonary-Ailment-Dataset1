import csv
import os
import re
import cv2
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        #print('self._base_dir----------------',self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

#############################血友病，数据集读取################################################
class BaseDataSets_xueyoubing(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        #print('self._base_dir----------------',self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "1-288.txt", "r") as f1:
            with open("/mnt/sdd/wjh/Xue_You_Bing/" + "585-1831.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "289-584.txt", "r") as f:
            with open("/mnt/sdd/wjh/Xue_You_Bing/" + "1-288.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # elif self.split == "test":
        #     with open(self._base_dir + "/test.list", "r") as f:
        #         self.sample_list = f.readlines()
        #     self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # 现在我有一个数据“1.jpg 0 0 0 0 0 0”，请你用Python帮我把这个数据一空格为分隔符号分成七个数据，第一个数据是字符形式的，之后的六个数据以整数的形式存放到一个列表中。
        split_data = case.split()
        filename = split_data[0]
        int_data = [int(value) for value in split_data[1:]]
        #print("int_data", int_data) int_data [0, 0, 0, 0, 0, 0]
        # if self.split == "train":
        #     h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        # else:
        #     h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        # 知道图片的路径，用Image读取为单通道的数据, 数据转成numpy格式的二位矩阵
        if self.split == "train":
            # _image = Image.open("/mnt/sdd/wjh/Xue_You_Bing/1-288/1-288/{}".format(filename))
            _image = Image.open("/mnt/sdd/wjh/Xue_You_Bing/585-1831/{}".format(filename))
        if self.split == "val":
            _image = Image.open("/mnt/sdd/wjh/Xue_You_Bing/1-288/1-288/{}".format(filename))
        gray_image = _image.convert("L")
        image = np.array(gray_image)
        label = int_data
        # sample = {"image": image, "label": label}
        sample = {"image": image}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["label"] = label
        sample["idx"] = idx
        return sample

#############################吸入综合征，数据集读取################################################
class BaseDataSets_Xi_Ru(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        #print('self._base_dir----------------',self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "1-288.txt", "r") as f1:
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/train.txt", "r") as f1:
            # with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/train1.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "289-584.txt", "r") as f:
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "289-584.txt", "r") as f:
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/test.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # elif self.split == "test":
        #     with open(self._base_dir + "/test.list", "r") as f:
        #         self.sample_list = f.readlines()
        #     self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # 现在我有一个数据“1.jpg 0 0 0 0 0 0”，请你用Python帮我把这个数据一空格为分隔符号分成七个数据，第一个数据是字符形式的，之后的六个数据以整数的形式存放到一个列表中。
        split_data = case.split()
        filename = split_data[0]
        int_data = int(split_data[1])
        #print("int_data", int_data) int_data [0, 0, 0, 0, 0, 0]
        # if self.split == "train":
        #     h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        # else:
        #     h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        # 知道图片的路径，用Image读取为单通道的数据, 数据转成numpy格式的二位矩阵
        if self.split == "train":
            if int_data == 0:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
            if int_data == 1:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Xi_Ru/{}".format(filename))
        if self.split == "val":
            if int_data == 0:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
            if int_data == 1:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Xi_Ru/{}".format(filename))
        if self.split == "test":
            if int_data == 0:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
            if int_data == 1:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Xi_Ru/{}".format(filename))
        # gray_image = _image.convert("L")
        # image = np.array(gray_image)
        image = np.array(_image)
        label = int_data
        # sample = {"image": image, "label": label}
        sample = {"image": image}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["label"] = label
        sample["idx"] = idx
        return sample

#############################吸入综合征，数据集读取，5折交叉验证################################################
class BaseDataSets_Xi_Ru_kfold(Dataset):
    def __init__(self, base_dir=None, split="train", fold=0, num=None, transform=None, ops_weak=None, ops_strong=None):
        """
        Args:
            base_dir: 存放 txt 的根目录，例如 /mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru
            split: "train", "val", 或 "test"
            fold: 当前折编号（0~4）
        """
        self._base_dir = base_dir
        self.fold = fold
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(ops_strong), \
            "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # === 1️⃣ 加载该 fold 对应的 txt 文件 ===
        txt_path = os.path.join(
            self._base_dir, "kfold_splits",
            f"{self.split}_fold{self.fold}.txt"
        )
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到文件: {txt_path}")

        with open(txt_path, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        print(f"[Fold {self.fold}] Loaded {self.split} samples: {len(self.sample_list)}")

        # === 2️⃣ 固定真实图片路径根目录 ===
        self.img_root = "/mnt/sdd/wjh/Fei_Bing/dataset_able"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        split_data = case.split()
        filename = split_data[0]
        int_data = int(split_data[1])

        # === 3️⃣ 这里保持你原来正确的逻辑 ===
        if self.split == "train":
            if int_data == 0:
                _image = Image.open(f"{self.img_root}/Normal_able/{filename}")
            elif int_data == 1:
                _image = Image.open(f"{self.img_root}/Xi_Ru/{filename}")

        elif self.split == "val":
            if int_data == 0:
                _image = Image.open(f"{self.img_root}/Normal_able/{filename}")
            elif int_data == 1:
                _image = Image.open(f"{self.img_root}/Xi_Ru/{filename}")

        elif self.split == "test":
            if int_data == 0:
                _image = Image.open(f"{self.img_root}/Normal_able/{filename}")
            elif int_data == 1:
                _image = Image.open(f"{self.img_root}/Xi_Ru/{filename}")

        else:
            raise ValueError(f"未知 split: {self.split}")

        image = np.array(_image, dtype=np.float32)
        image = image / 255.0  # 可选，但建议保留
        label = int_data

        sample = {"image": image, "label": label}
        if self.transform is not None:
            if self.split == "train" and None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        sample["label"] = label
        sample["idx"] = idx
        return sample
    
#############################吸入综合征，数据集读取,增加文本信息###############################################
class BaseDataSets_Xi_Ru_txt(Dataset):
    def __init__(self, base_dir=None, text_file=None, split="train", num=None, transform=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.text_data = {}

        # 确保同时提供弱增强和强增强策略
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # 加载图像数据集文件
        if self.split == "train":
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
        elif self.split == "val":
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/val.txt", "r") as f:
                self.sample_list = f.readlines()
        elif self.split == "test":
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/test.txt", "r") as f:
                self.sample_list = f.readlines()
        
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        print("Total {} samples loaded for {} split.".format(len(self.sample_list), self.split))

        # 加载xlsx格式的文本数据并构建放射编号对应关系
        if text_file is not None:
            df = pd.read_csv(text_file, dtype={'放射编号': str})
            for _, row in df.iterrows():
                radiology_id = str(row['放射编号'])  # 假设“放射编号”是Excel文件中的列名
                self.text_data[radiology_id] = {
                    '性别': row['性别'],
                    '年龄': row['年龄'],
                    '胎龄': row['胎龄'],
                    '出生体重': row['出生体重'],
                    '生产方式': row['生产方式'],
                    '羊水情况': row['羊水情况'],
                    '是否窒息': row['是否窒息']
                }
            print("Loaded {} text data entries from Excel.".format(len(self.text_data)))

        # 标准化器用于处理胎龄等数值数据
        self.scaler = StandardScaler()    

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        split_data = case.split()
        filename = split_data[0]  # 图像文件名，例如“151440.png”
        file_id = filename.split('.')[0]  # 提取图像文件名中的编号部分，例如“151440”
        file_id = file_id.zfill(6)

        # 根据编号找到对应的文本数据
        text_info = self.text_data.get(file_id, None)
        if text_info is None:
                    text_info = {
            '性别': '未知', 
            '年龄': 0, 
            '胎龄': 0.0, 
            '出生体重': 0.0, 
            '生产方式': '未知', 
            '羊水情况': '未知', 
            '是否窒息': '未知'
        }

        
        int_data = int(split_data[1])  # 图像类别
        if self.split == "train":
            if int_data == 0:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
            elif int_data == 1:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Xi_Ru/{}".format(filename))
        elif self.split == "val" or self.split == "test":
            if int_data == 0:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
            elif int_data == 1:
                _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Xi_Ru/{}".format(filename))
        
        image = np.array(_image)
        label = int_data


        # 将图像转换为Tensor格式
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # 返回图像、标签和文本数据的样本
        sample = {"image": image, "label": label, "text_info": text_info,"file_id":file_id}
        
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        sample["idx"] = idx
        sample["label"] = label
        sample["text_info"] = text_info
        sample["file_id"] = file_id

        return sample

#############################肺透明膜病5分类，数据集读取################################################
class BaseDataSets_Fei_Tou(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        #print('self._base_dir----------------',self._base_dir)
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "1-288.txt", "r") as f1:
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/train.txt", "r") as f1:
            # with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/train1.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "289-584.txt", "r") as f:
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            # with open("/mnt/sdd/wjh/Xue_You_Bing/" + "289-584.txt", "r") as f:
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/test.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # elif self.split == "test":
        #     with open(self._base_dir + "/test.list", "r") as f:
        #         self.sample_list = f.readlines()
        #     self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # 现在我有一个数据“1.jpg 0 0 0 0 0 0”，请你用Python帮我把这个数据一空格为分隔符号分成七个数据，第一个数据是字符形式的，之后的六个数据以整数的形式存放到一个列表中。
        split_data = case.split()
        filename = split_data[0]
        int_data = int(split_data[1])
        #print("int_data", int_data) int_data [0, 0, 0, 0, 0, 0]
        # if self.split == "train":
        #     h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        # else:
        #     h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        # 知道图片的路径，用Image读取为单通道的数据, 数据转成numpy格式的二位矩阵
        
        if int_data == 0:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
        if int_data == 1:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_1/{}".format(filename))
        if int_data == 2:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_2/{}".format(filename))
        if int_data == 3:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_3/{}".format(filename))
        if int_data == 4:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_4/{}".format(filename))

        
        gray_image = _image.convert("L")
        image = np.array(gray_image)
        label = int_data
        # sample = {"image": image, "label": label}
        sample = {"image": image}

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["label"] = label
        sample["idx"] = idx
        return sample

#############################肺透明膜病5分类，数据集读取，5折交叉验证################################################    
class BaseDataSets_Fei_Tou_kfold(Dataset):
    """
    Fei_Tou（肺透）任务的 K-fold 数据集类
    与 Xi_Ru（吸入）版本完全一致的 k折结构
    """

    def __init__(self, base_dir=None, split="train", fold=0, num=None,
                 transform=None, ops_weak=None, ops_strong=None):
        """
        Args:
            base_dir: 存放 txt 的根目录，例如
                      /mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou
            split: "train", "val", "test"
            fold: 当前折编号（0~4）
        """
        self._base_dir = base_dir
        self.fold = fold
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(ops_strong), \
            "For CTAugment, both ops_weak and ops_strong must be provided together"

        # === 1️⃣ 加载 k-fold 对应的 txt 文件 ===
        txt_path = os.path.join(
            self._base_dir, "kfold_splits",
            f"{self.split}_fold{self.fold}.txt"
        )
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到文件: {txt_path}")

        with open(txt_path, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        print(f"[Fei_Tou Fold {self.fold}] Loaded {self.split} samples: {len(self.sample_list)}")

        # === 2️⃣ 真实图像根目录 ===
        self.img_root = "/mnt/sdd/wjh/Fei_Bing/dataset_able"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        split_data = case.split()

        filename = split_data[0]
        label = int(split_data[1])   # 0–4 五类

        # === 3️⃣ 根据 label 加载图像（保持你原来的逻辑） ===
        if label == 0:
            path = f"{self.img_root}/Normal_able/{filename}"
        elif label == 1:
            path = f"{self.img_root}/Fei_Tou_1/{filename}"
        elif label == 2:
            path = f"{self.img_root}/Fei_Tou_2/{filename}"
        elif label == 3:
            path = f"{self.img_root}/Fei_Tou_3/{filename}"
        elif label == 4:
            path = f"{self.img_root}/Fei_Tou_4/{filename}"
        else:
            raise ValueError(f"未知 label: {label}")

        _image = Image.open(path)

        # 转灰度图
        gray_image = _image.convert("L")
        image = np.array(gray_image, dtype=np.float32)
        image = image / 255.0  # 建议保留

        # sample 格式保持一致
        sample = {"image": image, "label": label}

        # === 4️⃣ transform & CTAugment 支持 ===
        if self.transform is not None:
            if self.split == "train" and None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        # === 5️⃣ 额外信息 ===
        sample["idx"] = idx

        return sample

#############################肺透明膜病5分类，数据集读取,增加文本信息################################################
class BaseDataSets_Fei_Tou_txt(Dataset):
    def __init__(self, base_dir=None, split="train", text_file=None,num=None, transform=None, ops_weak=None, ops_strong=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.text_data = {}        

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/train2.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open("/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Fei_Tou/test.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {} samples".format(len(self.sample_list)))

        # 加载xlsx格式的文本数据并构建放射编号对应关系
        if text_file is not None:
            df = pd.read_csv(text_file, dtype={'编号': str})
            for _, row in df.iterrows():
                radiology_id = str(row['编号'])  # 假设“放射编号”是Excel文件中的列名
                self.text_data[radiology_id] = {
                    '性别': row['性别'],
                    '年龄': row['年龄'],
                    '胎龄': row['胎龄'],
                    '体重': row['体重'],
                    '生产方式': row['生产方式'],
                    '羊水情况': row['羊水情况'],
                    '是否窒息史': row['是否窒息史']
                }
            print("Loaded {} text data entries from Excel.".format(len(self.text_data)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        split_data = case.split()
        filename = split_data[0]
        file_id = filename.split('.')[0]  # 提取图像文件名中的编号部分，例如“151440”
        file_id = file_id.zfill(6)

        # print(f"Sample list length: {len(self.sample_list)}")


        # 根据编号找到对应的文本数据
        text_info = self.text_data.get(file_id, None)
        if text_info is None:
            text_info = {
               '性别': ['女'],  # 假设默认性别为'女'
               '年龄': ['0小时'],  # 默认年龄为0小时
               '胎龄': ['25+0周'],  # 默认胎龄为0周
               '体重': ['0g'],  # 默认体重为0g
               '生产方式': ['顺产'],  # 默认生产方式为顺产
               '羊水情况': ['清'],  # 假设默认羊水情况为清
               '是否窒息史': ['是']  # 假设默认窒息史为否
           }
            print("text_info,file_id",text_info,file_id)

        int_data = int(split_data[1])
        
        if int_data == 0:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Normal_able/{}".format(filename))
        if int_data == 1:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_1/{}".format(filename))
        if int_data == 2:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_2/{}".format(filename))
        if int_data == 3:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_3/{}".format(filename))
        if int_data == 4:
            _image = Image.open("/mnt/sdd/wjh/Fei_Bing/dataset_able/Fei_Tou_4/{}".format(filename))

        
        gray_image = _image.convert("L")
        image = np.array(gray_image)
        label = int_data

        sample = {"image": image, "label": label, "text_info": text_info,"file_id":file_id}

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        sample["label"] = label
        sample["text_info"] = text_info
        sample["file_id"] = file_id
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def random_rotate_single(image):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    return image

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
class RandomGenerator_single(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image = random_rot_flip(image)
        elif random.random() > 0.5:
            image = random_rotate_single(image)  
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        
        sample = {"image": image}
        return sample

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
