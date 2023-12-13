from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from functools import partial
from tqdm import tqdm



class KittiDataset(Dataset):
    def __init__(self, Img_Dir, Mask_Dir, Img_Size, scale):
        
        self.Img_Dir = Img_Dir
        self.Mask_Dir = Mask_Dir
        self.scale = scale
        self.Img_Size = Img_Size
        self.ids = [splitext(file)[0] for file in listdir(Img_Dir) if not file.startswith('.')]
        

    @classmethod    
    def preprocessing(cls, pImg, pSize, scale, is_mask):
        reImg = pImg.resize((pSize[1], pSize[0]), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        w, h = reImg.size
        newW, newH = int(scale * w), int(scale * h)
        
        pil_img = pImg.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        img_ndarray = np.asarray(pil_img)
        
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))        
        
        if not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
        
        if is_mask:
            img_ndarray = img_ndarray.astype('float32')
            img_ndarray /= 255.0
        
        

        return img_ndarray
    
    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        ## if npz, npy case
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        ## if pt, pth
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        ## if image
        else:
            return Image.open(filename)
    
    
    def __getitem__(self, idx):
        
        name = self.ids[idx]
        
        ## 2023-12-12
        #img_file = list(self.Img_Dir.glob(name + '*.png'))  
        #mask_file = list(self.Mask_Dir.glob(name + '*.png'))     
        #img = Image.open(img_file[0])
        #mask = Image.open(mask_file[0])
        
        # 이거 이렇게 하면 resize가 문제가 됨 고쳐야함
        mask_file = list(self.Mask_Dir.glob(name + '.*'))
        img_file = list(self.Img_Dir.glob(name + '.*'))
        
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        
        
        img = self.preprocessing(img, self.Img_Size, self.scale, is_mask=False)
        mask = self.preprocessing(mask, self.Img_Size, self.scale, is_mask=True)   
        
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.ids)