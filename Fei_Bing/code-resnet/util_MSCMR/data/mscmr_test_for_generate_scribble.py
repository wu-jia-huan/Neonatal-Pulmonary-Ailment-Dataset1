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
from skimage.morphology import skeletonize, dilation, closing
import cv2
import copy

def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header
def generate_skeleton_scribble(mask):
    """ Scribbles are approximated by a skeleton of the image
    :param mask: multi-channel binary mask
    :return: scribbles
    """
    # initialize scribbles as empty array
    scribbles = np.zeros_like(mask)
    n_channels = mask.shape[-1]

    for ch in range(n_channels):
        # extract skeleton from the current channel
        m = np.copy(mask[:, :, ch])
        skl = skeletonize(m)

        # make slightly thicker (but always inside the gt mask)
        skl = closing(skl)
        skl = dilation(skl) * m

        # assign skeleton to return array
        scribbles[..., ch] = skl

    return scribbles
def _per_class_random_walk(mask, length_coeff=None):
    """ Generate smooth (self-avoiding) random walk for each class"""

    if length_coeff is None:
        length_coeff = [0.10, 0.10, 0.10, 0.10]
    assert len(mask.shape) == 3  # 2D + class

    W, H, C = mask.shape
    walk_lengths = [int(length_coeff[i] * np.sum(mask[..., i])) for i in range(C)]

    # initialize 3D mask of random walks
    random_walks = np.zeros_like(mask)

    n_channels = mask.shape[-1]
    for ch in range(n_channels):
        m = mask[:, :, ch]

        # get position of pixels belonging to the mask
        where = np.argwhere(m)

        if walk_lengths[ch] >= len(where):
            random_walks[..., ch] = m
        else:
            # initialize 2D mask of zeros to walk in
            _m = np.zeros_like(m)

            # get random seed and initialize to 1
            seed_x, seed_y = where[np.random.randint(0, len(where))]
            _m[seed_x, seed_y] = 1

            last_x, last_y = None, None
            max_iters = 4000
            while len(np.argwhere(_m)) < walk_lengths[ch] and max_iters > 0:
                max_iters -= 1
                x = seed_x + np.random.choice([1, 0, -1])
                y = seed_y + np.random.choice([1, 0, -1])

                if x < 0: x = 0
                if y < 0: y = 0
                if x > W - 1: x = W - 1
                if y > W - 1: y = H - 1

                if not (last_x is None and last_y is None):
                    if last_x != x and last_y != y:
                        # If we are inside the mask, assign 1 to the random walk
                        if m[x, y] == 1:
                            _m[x, y] = 1
                            last_x, last_y = seed_x, seed_y
                            seed_x, seed_y = x, y
                else:
                    # If we are inside the mask, assign 1 to the random walk
                    if m[x, y] == 1:
                        _m[x, y] = 1
                        last_x, last_y = seed_x, seed_y
                        seed_x, seed_y = x, y

            # smooth the random walk:
            _m = closing(dilation(_m))
            _m = skeletonize(_m)

            # make thick (always inside the gt mask)
            _m = dilation(_m) * m

            # assign random walk to the current channel
            random_walks[..., ch] = _m

    return random_walks
##################################
# build_gt_to_scribble_dataset 是精标签转换成涂鸦标注的，generate_method是生成涂鸦标注的办法
def build_gt_to_scribble_dataset(generate_method, image_set, args, label_ratio, labeled_type):
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
        dataset = mscmrSeg_gt_to_scribble_dataset(generate_method, label_ratio, labeled_type, image_set, img_task, lab_task, lab_values, transforms=transforms.Compose([RandomGenerator([256, 256])]))
        # dataset_dict.update({task : dataset})
        # print('dataset_dict', dataset_dict.items)
    #return dataset_dict
    return dataset
#精标签监督带标签比例的
class mscmrSeg_gt_to_scribble_dataset(data.Dataset):
    def __init__(self,generate_method, label_ratio, labeled_type, image_set, img_folder, lab_folder, lab_values, transforms):
        self.generate_method = generate_method
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
            
            #在这一步直接转化精标签为涂鸦标注
            H, W = lab.shape
            scribble_lab_final = np.zeros_like(lab)
            lab_3c = np.zeros((H, W, 3))
            #一化三，三变三，三合一
            lab_slice_0 = copy.deepcopy(lab)
            lab_slice_0[lab_slice_0==600] = 1
            lab_slice_0[lab_slice_0==200] = 0
            lab_slice_0[lab_slice_0==500] = 0

            lab_slice_1 = copy.deepcopy(lab)
            lab_slice_1[lab_slice_1==600] = 0
            lab_slice_1[lab_slice_1==200] = 1
            lab_slice_1[lab_slice_1==500] = 0


            lab_slice_2 = copy.deepcopy(lab)
            lab_slice_2[lab_slice_2==600] = 0
            lab_slice_2[lab_slice_2==200] = 0
            lab_slice_2[lab_slice_2==500] = 1
            
            lab_3c[:, :, 0] = lab_slice_0
            lab_3c[:, :, 1] = lab_slice_1
            lab_3c[:, :, 2] = lab_slice_2
            
            #三变三
            if self.generate_method =='sk':
                scribble_lab = generate_skeleton_scribble(lab_3c)
            if self.generate_method =='rw':
                scribble_lab = _per_class_random_walk(lab_3c, [0.9, 0.9, 0.9])
            
            #三合一
            scribble_lab[:, :, 0][scribble_lab[:, :, 0] == 1] = 1
            scribble_lab[:, :, 1][scribble_lab[:, :, 1] == 1] = 2
            scribble_lab[:, :, 2][scribble_lab[:, :, 2] == 1] = 3
            
            scribble_lab_final = scribble_lab[:, :, 0] + scribble_lab[:, :, 1] + scribble_lab[:, :, 2]
            scribble_lab_final_unique = np.unique(scribble_lab_final)
            print('scribble_lab_final_unique-----------------',scribble_lab_final_unique)
            
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
            sample = {'image': img, 'label': scribble_lab_final}
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
    
