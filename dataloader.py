import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
class dataload(Dataset):
    def __init__(self,train:bool):
        self.train=train
        self.data_type=input("enter type of data for testing or training: ")
        if self.data_type=='type_1':
            self.data_path='/home/govind/data/type_1/type1_data'
        elif self.data_type=='type_3':
            self.data_path='/home/govind/data/type_3/type3_data'
        elif self.data_type=='type_5':
            self.data_path='/home/govind/data/type_5/type5_data'
        elif self.data_type=='type_7':
            self.data_path='/home/govind/data/type_7/type7_data'
        elif self.data_type=='type_9':
            self.data_path='/home/govind/data/type_9/type9_data'
        elif self.data_type=='type1':
            self.data_path='/home/govind/data/type1/typeI_data'
        elif self.data_type=='type1A':
            self.data_path='/home/govind/data/typeIA/typeIA_data'
        elif self.data_type=='type1B':
            self.data_path='/home/govind/data/typeIB/typeIB_data'
        elif self.data_type=='typeII':
            self.data_path='/home/govind/data/typeII/typeII_data/typeII_data'
        elif self.data_type=='typeIII':
            self.data_path='/home/govind/data/typeIII/typeIII_data'
        else:
            raise Exception("Enter valid data type")
        lisdir=os.listdir(self.data_path)
        self.raw_img_fld_path=os.path.join(self.data_path,'underwater_typeI')
        self.gt_img_fld_path=os.path.join(self.data_path,'gt_typeI')




    def __len__(self):
        if (self.train):
            return 1000
        else:
            return 449
    def __getitem__(self, index):
        if (self.train):
            st_idx=0
        else:
            st_idx=1000
        raw_image_lsdir=os.listdir(self.raw_img_fld_path)
        gt_image_lsdir=os.listdir(self.gt_img_fld_path)
        raw_image=Image.open(os.path.join(self.raw_img_fld_path,raw_image_lsdir[st_idx+index]))
        gt_image=Image.open(os.path.join(self.gt_img_fld_path,gt_image_lsdir[st_idx+index]))
        raw_image=transforms.Resize(230)(raw_image)
        gt_image=transforms.Resize(230)(gt_image)
        raw_image=torch.tensor(np.array(raw_image)).permute(2,1,0).float()
        gt_image=torch.tensor(np.array(gt_image)).permute(2,1,0).float()
        return raw_image,gt_image
