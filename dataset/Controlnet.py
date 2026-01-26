
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import torchio as tio


class ConditionalDataset(Dataset):
    def __init__(self, condition_dir=None, gt_dir=None ,ref_dir=None,split='train', resolution=[16,64,64], filter_by_resolution=False):
        self.condition_dir = condition_dir
        self.ref_dir = ref_dir
        self.split = split
        self.resolution = resolution
        self.filter_by_resolution = filter_by_resolution

        self.condition_names = glob.glob(os.path.join(
            condition_dir, './*.nii.gz'), recursive=True)
        self.gt_names = glob.glob(os.path.join(
            gt_dir, './*.nii.gz'), recursive=True)
        self.condition_names.sort()
        self.gt_names.sort()

        # Filter by resolution if requested
        if self.filter_by_resolution:
            filtered_condition = []
            filtered_gt = []
            target_shape = (8, self.resolution[0], self.resolution[1], self.resolution[2])
            for cond_path, gt_path in zip(self.condition_names, self.gt_names):
                img = tio.ScalarImage(cond_path)
                if img.shape == target_shape:
                    filtered_condition.append(cond_path)
                    filtered_gt.append(gt_path)
            print(f"Filtered {len(self.condition_names)} → {len(filtered_condition)} files matching shape {target_shape}")
            self.condition_names = filtered_condition
            self.gt_names = filtered_gt

        self.split = split
        if split == 'train':
            self.condition_names = self.condition_names[:-40]
            self.gt_names = self.gt_names[:-40]

        elif split == 'val':
            self.condition_names = self.condition_names[-40:]
            self.gt_names = self.gt_names[-40:]
        else:
            self.condition_names = self.condition_names[:]
            self.gt_names = self.gt_names[:]


    def __len__(self):

        return len(self.condition_names)

    def __getitem__(self, index):
        if self.split != 'test':
            condition_path = self.condition_names[index]
            gt_path = self.gt_names[index]
            condition_img ,gt_img = tio.ScalarImage(condition_path) , tio.ScalarImage(gt_path)

            data_con, data_gt = condition_img.data , gt_img.data
            data_con, data_gt = data_con.to(torch.float32), data_gt.to(torch.float32)
        else:
            condition_path = self.condition_names[index]
            condition_img = tio.ScalarImage(condition_path) 
            data_con = condition_img.data 
            data_con = data_con.to(torch.float32)


        
        if self.split =='val':
            ref_path = os.path.join(self.ref_dir,os.path.basename(condition_path))
            ref_img = tio.ScalarImage(ref_path)
            affine = ref_img.affine
            return {'data': data_con, 'gt':data_gt ,'affine' : affine , 'path':condition_path , 'y':0}
        elif self.split =='test':
            ref_path = os.path.join(self.ref_dir,os.path.basename(condition_path))
            ref_img = tio.ScalarImage(ref_path)
            affine = ref_img.affine
            return {'data': data_con, 'affine' : affine , 'path':condition_path, 'y':0}
        else:
            return {'data': data_con, 'gt':data_gt,'y':0}
