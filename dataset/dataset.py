import logging
import re

import numpy as np
import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

join = os.path.join
listdir = os.listdir
splitext = os.path.splitext


class BaseDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform, target_transform, scale=1, mask_suffix='', Base=True):
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.target_transform = target_transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        if Base == True:
            self.imgs_dir = imgs_dir
            self.masks_dir = masks_dir
            self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Create dataset with len(self.ids) example')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newH > 0 and newW > 0, f'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        # ensure img_nd is three dim as (h, w, c)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # transpose to c, h ,w
        # img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_nd
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans.astype(np.float64)

    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_name = self.masks_dir + idx + self.mask_suffix + '.*'
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        # ensure only one file
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # img.convert('RGB')
        assert img.size == mask.size, f'Image and mask {idx} should be the same size'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # return self.transform(img).float(), self.target_transform(mask).float()

        return {
            # 'image': torch.from_numpy(img).type(torch.FloatTensor),
            # 'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            'image': self.transform(img).float(),
            'mask': self.target_transform(mask).float()
        }


class ChaoDataset(BaseDataset):
    def __init__(self, imgs_dir, masks_dir, transform, target_transform, scale=1, mask_suffix='', type='CT'):
        super(ChaoDataset, self).__init__(imgs_dir, masks_dir, transform, target_transform, scale, mask_suffix,
                                          Base=False)
        self.type = type
        self.imgs_dir_root = imgs_dir
        self.masks_dir_root = masks_dir
        self.ids = []
        self.mask_ids = []
        if type == 'CT':
            path = self.imgs_dir_root + 'CT/'
            for case in os.listdir(path):
                dcm_data = path + case + '/CTdcm2png/'
                label_data = path + case + '/Ground/'
                # diff_part = 'CT/' + case
                dcm_list = listdir(dcm_data)
                re_pattern = re.compile(r'(\d+)')
                label_list = listdir(label_data)
                dcm_list.sort(key=lambda x: re_pattern.findall(x)[-1] + re_pattern.findall(x)[0])
                label_list.sort(key=lambda x: re_pattern.findall(x)[-1])
                for file in dcm_list:
                    result = dcm_data + file[:-4]
                    self.ids.append(result)

                    # label_result = label_data + 'liver_GT_' + file[-7:-4]
                    label_result = label_data + label_list[dcm_list.index(file)][:-4]
                    self.mask_ids.append(label_result)
                # ids = [dcm_data + splitext(file)[0] for file in listdir(dcm_data) if
                #        not file.startswith('.')]
                # mask_ids = [label_data + splitext(file)[0] for file in listdir(label_data) if
                #             not file.startswith('.')]
                # self.ids.extend(ids)
                # self.mask_ids.extend(mask_ids)

                # save_path = path + case + '/CTdcm2png/'
                # for img in os.listdir(dcm_data):
                # multi_dcm(dcm_data, save_path)
        elif type == 'MR':
            path = self.imgs_dir_root + 'MR/'
            for case in os.listdir(path):
                dcm_data = path + case + '/T2SPIR/MRdcm2png/'
                label_data = path + case + '/T2SPIR/Ground/'
                ids = [dcm_data + splitext(file)[0] for file in listdir(dcm_data) if not file.startswith('.')]
                mask_ids = [label_data + splitext(file)[0] for file in listdir(label_data) if
                            not file.startswith('.')]
                self.ids.extend(ids)
                self.mask_ids.extend(mask_ids)
                # save_path = path + case + '/T2SPIR/MRdcm2png/'
                # for img in os.listdir(dcm_data):
                # multi_dcm(dcm_data, save_path)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_idx = self.mask_ids[i]
        # mask_name = self.masks_dir + idx + self.mask_suffix + '.*'
        a = mask_idx + self.mask_suffix + '.*'
        mask_file = glob(mask_idx + self.mask_suffix + '.*')
        img_file = glob(idx + '.*')
        # ensure only one file
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        # mask.show()
        # img.show()
        # img.convert('RGB')
        assert img.size == mask.size, f'Image and mask {idx} should be the same size'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        # return self.transform(img).float(), self.target_transform(mask).float()

        return {
            # 'image': torch.from_numpy(img).type(torch.FloatTensor),
            # 'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            'image': self.transform(img).float(),
            'mask': self.target_transform(mask).float()
        }


def train_dataloader(train_data_list, batch_size, ar=None):
    """

    :param ar:
    :param train_data_list:
    :param batch_size:
    :return: Dataloader
    """


    # if ar is None:
    #     result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=True, num_workers=16,
    #                         pin_memory=True)
    # else:
    #     result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=False, num_workers=16,
    #                         pin_memory=True, drop_last=True)
    # return result



    if ar is None:
        result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=True, num_workers=4,
                            pin_memory=True)
    else:
        result = DataLoader(dataset=train_data_list, batch_size=batch_size, shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=True)
    return result
