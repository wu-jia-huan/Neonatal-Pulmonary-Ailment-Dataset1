import SimpleITK as sitk
import numpy as np
from pathlib import Path

import torch
from torch.utils import data
import nibabel as nib

import util_MSCMR.data.transforms as T

#add
from torchvision import transforms
from util_MSCMR.data.transforms import RandomGenerator
from scipy.ndimage.interpolation import zoom
from dataloaders.dataset import WeakStrongAugment


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

class mscmrSeg(data.Dataset):
    def __init__(self, image_set, img_folder, lab_folder, lab_values, transforms):
        self.image_set = image_set
        self.transforms = transforms
        img_paths = list(img_folder.iterdir())
        lab_paths = list(lab_folder.iterdir())
        self.lab_values = lab_values
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict.update({img_name : img})
            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            print(img_name, lab_name)
            self.lab_dict.update({lab_name : lab})
            # self.examples += [(img_name, lab_name, slice, -1, -1) for slice in range(img.shape[0])]
            #assert img.shape[1] == lab.shape[1]
            #self.examples += [(img_name, lab_name, -1, slice, -1) for slice in range(img.shape[1])]
            assert img[0].shape[2] == lab[0].shape[2]
            self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
            
    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]
        if Z != -1:
            img = self.img_dict[img_name][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        # img = np.expand_dims(img, 0)
        # lab = np.expand_dims(lab, 0)
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
        #if self._transforms is not None:
            #img, target = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
            #img, lab = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
        
        if self.image_set == 'train':
            sample = {'image': img, 'label': lab}
            sample = self.transforms(sample)
        if self.image_set == 'val' or self.image_set == 'test':
            image = torch.from_numpy(
                img.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(lab.astype(np.uint8)).unsqueeze(0)
            sample = {'image': image, 'label': label}
        # print('img size   ',sample['image'].shape)
        # print('lab size   ',sample['label'].shape)
        return sample

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img-img.mean())/img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)
    
def make_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        return T.Compose([
            T.Rescale(),
            T.RandomHorizontalFlip(),
            T.RandomRotate((0,360)),
            T.PadOrCropToSize([212,212]),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([ 
            T.Rescale(),
            T.PadOrCropToSize([212,212]),
            normalize])


    raise ValueError(f'unknown {image_set}')

#直接build是全部弱标签,但是val和test就直接build就好
def build(image_set, args):
    # set your data path
    root = Path(args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "val" / "images", root / "val" / "labels"),
        "test": (root / "TestSet" / "images", root / "TestSet" / "labels"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg(image_set, img_task, lab_task, lab_values, transforms=transforms.Compose([RandomGenerator([256, 256])]))
        # dataset_dict.update({task : dataset})
        # print('dataset_dict', dataset_dict.items)
    #return dataset_dict
    return dataset

#build_weakly_ratio_dataset是弱监督带标签比例的,调用的是mscmrSeg_weakly_ratio
def build_weakly_ratio_dataset(image_set, args, label_ratio, labeled_type):
    # set your data path
    root = Path(args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "val" / "images", root / "val" / "labels"),
        "test": (root / "TestSet" / "images", root / "TestSet" / "labels"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg_weakly_ratio(label_ratio, labeled_type, image_set, img_task, lab_task, lab_values, transforms=transforms.Compose([RandomGenerator([256, 256])]))
        # dataset_dict.update({task : dataset})
        # print('dataset_dict', dataset_dict.items)
    #return dataset_dict
    return dataset

#弱监督带标签比例的
class mscmrSeg_weakly_ratio(data.Dataset):
    def __init__(self, label_ratio, labeled_type, image_set, img_folder, lab_folder, lab_values, transforms):
        self.label_ratio = label_ratio
        self.labeled_type = labeled_type
        self.image_set = image_set
        self.transforms = transforms
        img_paths = list(img_folder.iterdir())
        img_paths = sorted(img_paths)
        lab_paths = list(lab_folder.iterdir())
        lab_paths = sorted(lab_paths)
        
        len_img_paths = len(img_paths)
        len_lab_paths = len(lab_paths)
        print('1------len_img_paths',len_img_paths)
        print(img_paths)
        print('2------len_lab_paths',len_lab_paths)
        print(lab_paths)
        self.lab_values = lab_values
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        if self.labeled_type == 'labeled':
            img_paths = img_paths[:int(len_img_paths * label_ratio + 0.5)]
            lab_paths = lab_paths[:int(len_lab_paths * label_ratio + 0.5)]
            # print('3------labeled_img',img_paths)
            # print(img_paths)
            # print('3------labeled_lab',lab_paths)
            # print(lab_paths)
        elif self.labeled_type == 'unlabeled':
            img_paths = img_paths[int(len_img_paths * label_ratio + 0.5):]
            lab_paths = lab_paths[int(len_lab_paths * label_ratio + 0.5):]
            # print('4------unlabeled_img',img_paths)
            # print(img_paths)
            # print('4------unlabeled_lab',lab_paths)
            # print(lab_paths)
        else:
            raise Exception("labeled_type error!")
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict.update({img_name : img})
            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            print(img_name, lab_name)
            self.lab_dict.update({lab_name : lab})
            # self.examples += [(img_name, lab_name, slice, -1, -1) for slice in range(img.shape[0])]
            #assert img.shape[1] == lab.shape[1]
            #self.examples += [(img_name, lab_name, -1, slice, -1) for slice in range(img.shape[1])]
            assert img[0].shape[2] == lab[0].shape[2]
            self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
            
    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]
        if Z != -1:
            img = self.img_dict[img_name][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        # img = np.expand_dims(img, 0)
        # lab = np.expand_dims(lab, 0)
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
        #if self._transforms is not None:
            #img, target = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
            #img, lab = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
        
        if self.image_set == 'train':
            sample = {'image': img, 'label': lab}
            sample = self.transforms(sample)
        if self.image_set == 'val' or self.image_set == 'test':
            image = torch.from_numpy(
                img.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(lab.astype(np.uint8)).unsqueeze(0)
            sample = {'image': image, 'label': label}
        return sample

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img-img.mean())/img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)
###################################################################################################
# build_gt_dataset 是全部的精标签
def build_gt_dataset(image_set, args, label_ratio, labeled_type):
    # set your data path
    root = Path(args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels_exact"),
        "val": (root / "val" / "images", root / "val" / "labels"),
        "test": (root / "TestSet" / "images", root / "TestSet" / "labels"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg_gt_dataset(label_ratio, labeled_type, image_set, img_task, lab_task, lab_values, transforms=transforms.Compose([RandomGenerator([256, 256])]))
        # dataset_dict.update({task : dataset})
        # print('dataset_dict', dataset_dict.items)
    #return dataset_dict
    return dataset
#build_gt_dataset_cross_teacher是因为cross teacher中用256,256会报错尺寸不对，所以改成这个了
def build_gt_dataset_cross_teacher(image_set, args, label_ratio, labeled_type):
    # set your data path
    root = Path(args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels_exact"),
        "val": (root / "val" / "images", root / "val" / "labels"),
        "test": (root / "TestSet" / "images", root / "TestSet" / "labels"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg_gt_dataset(label_ratio, labeled_type, image_set, img_task, lab_task, lab_values, transforms=transforms.Compose([RandomGenerator([224, 224])]))
    return dataset
#build_gt_dataset_fixmatch是因为fixmatch这个半监督方法的transform需要一个特殊的。
def build_gt_dataset_fixmatch(image_set, args, label_ratio, labeled_type):
    # set your data path
    root = Path(args.dataset)
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels_exact"),
        "val": (root / "val" / "images", root / "val" / "labels"),
        "test": (root / "TestSet" / "images", root / "TestSet" / "labels"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg_gt_dataset(label_ratio, labeled_type, image_set, img_task, lab_task, lab_values, transforms=transforms.Compose([WeakStrongAugment([256, 256])]))
        # dataset_dict.update({task : dataset})
        # print('dataset_dict', dataset_dict.items)
    #return dataset_dict
    return dataset
#精标签监督带标签比例的
class mscmrSeg_gt_dataset(data.Dataset):
    def __init__(self, label_ratio, labeled_type, image_set, img_folder, lab_folder, lab_values, transforms):
        self.label_ratio = label_ratio
        self.labeled_type = labeled_type
        self.image_set = image_set
        self.transforms = transforms
        img_paths = list(img_folder.iterdir())
        img_paths = sorted(img_paths)
        lab_paths = list(lab_folder.iterdir())
        lab_paths = sorted(lab_paths)
        
        len_img_paths = len(img_paths)
        len_lab_paths = len(lab_paths)
        print('1------len_img_paths',len_img_paths)
        print(img_paths)
        print('2------len_lab_paths',len_lab_paths)
        print(lab_paths)
        self.lab_values = lab_values
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}
        if self.labeled_type == 'labeled':
            img_paths = img_paths[:int(len_img_paths * label_ratio + 0.5)]
            lab_paths = lab_paths[:int(len_lab_paths * label_ratio + 0.5)]
            # print('3------labeled_img',img_paths)
            # print(img_paths)
            # print('3------labeled_lab',lab_paths)
            # print(lab_paths)
        elif self.labeled_type == 'unlabeled':
            img_paths = img_paths[int(len_img_paths * label_ratio + 0.5):]
            lab_paths = lab_paths[int(len_lab_paths * label_ratio + 0.5):]
            # print('4------unlabeled_img',img_paths)
            # print(img_paths)
            # print('4------unlabeled_lab',lab_paths)
            # print(lab_paths)
        else:
            raise Exception("labeled_type error!")
        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict.update({img_name : img})
            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            print(img_name, lab_name)
            self.lab_dict.update({lab_name : lab})
            # self.examples += [(img_name, lab_name, slice, -1, -1) for slice in range(img.shape[0])]
            #assert img.shape[1] == lab.shape[1]
            #self.examples += [(img_name, lab_name, -1, slice, -1) for slice in range(img.shape[1])]
            assert img[0].shape[2] == lab[0].shape[2]
            self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]
            
    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]
        if Z != -1:
            img = self.img_dict[img_name][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            lab = self.lab_dict[lab_name][0][:, :, Y]
            #值的转换 0-》4, 200-》2， 500-》3， 600-》1
            lab[lab==200] = 2
            lab[lab==500] = 3
            lab[lab==600] = 1
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        # img = np.expand_dims(img, 0)
        # lab = np.expand_dims(lab, 0)
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}
        #if self._transforms is not None:
            #img, target = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
            #img, lab = self._transforms([img, scale_vector_img], [target,scale_vector_lab])
        
        if self.image_set == 'train':
            sample = {'image': img, 'label': lab}
            sample = self.transforms(sample)
        if self.image_set == 'val' or self.image_set == 'test':
            image = torch.from_numpy(
                img.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(lab.astype(np.uint8)).unsqueeze(0)
            sample = {'image': image, 'label': label}
        return sample

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img-img.mean())/img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        # cla = np.asarray([(lab == v)*i for i, v in enumerate(self.lab_values)], np.int32)
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)
    


# #WeakStrongAugment是fixmatch中需要用到的一个变换方法
# class WeakStrongAugment(object):
#     """returns weakly and strongly augmented images

#     Args:
#         object (tuple): output size of network
#     """

#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample["image"], sample["label"]
#         image = self.resize(image)
#         label = self.resize(label)
#         # weak augmentation is rotation / flip
#         image_weak, label = random_rot_flip(image, label)
#         # strong augmentation is color jitter
#         image_strong = color_jitter(image_weak).type("torch.FloatTensor")
#         # fix dimensions
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.uint8))

#         sample = {
#             "image": image,
#             "image_weak": image_weak,
#             "image_strong": image_strong,
#             "label_aug": label,
#         }
#         return sample

#     def resize(self, image):
#         x, y = image.shape
#         return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
# def random_rot_flip(image, label=None):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     if label is not None:
#         label = np.rot90(label, k)
#         label = np.flip(label, axis=axis).copy()
#         return image, label
#     else:
#         return image


# def random_rotate(image, label):
#     angle = np.random.randint(-20, 20)
#     image = ndimage.rotate(image, angle, order=0, reshape=False)
#     label = ndimage.rotate(label, angle, order=0, reshape=False)
#     return image, label


# def color_jitter(image):
#     if not torch.is_tensor(image):
#         np_to_tensor = transforms.ToTensor()
#         image = np_to_tensor(image)

#     # s is the strength of color distortion.
#     s = 1.0
#     #判断的时候说这里要三通道，彩色的图像进行颜色抖动当然有亮度、对比度、饱和度、色相
#     #但是对于单通达的图像来说，只有亮度这一个属性可以变化brightness
#     #jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     jitter = transforms.ColorJitter(0.8 * s)
#     return jitter(image)

