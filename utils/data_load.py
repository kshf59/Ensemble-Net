from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

def unique_mask_values(idx, mask_dir):
    #mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask_file = list(mask_dir.glob(name + '*.png'))[0]
    mask = np.asarray(Image.open(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
        

class KittiDataset(Dataset):
    def __init__(self, Img_Dir, Mask_Dir, Img_Size, scale):
        
        self.Img_Dir = Img_Dir
        self.Mask_Dir = Mask_Dir
        self.scale = scale
        self.Img_Size = Img_Size
        self.ids = [splitext(file)[0] for file in listdir(Img_Dir) if not file.startswith('.')]
        
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.Mask_Dir), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
    
    @staticmethod    
    def preprocessing(mask_values, pImg, pSize, scale, is_mask):
        reImg = pImg.resize((pSize[1], pSize[0]), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        w, h = reImg.size
        newW, newH = int(scale * w), int(scale * h)
        
        pil_img = pImg.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        img_ndarray = np.asarray(pil_img)
        
        '''
        if not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
        
        if is_mask:
            img_ndarray = img_ndarray.astype('float32')
            img_ndarray /= 255.0
        '''
        
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img_ndarray.ndim == 2:
                    mask[img_ndarray == v] = i
                else:
                    mask[(img_ndarray == v).all(-1)] = i

            return mask

        else:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            if (img_ndarray > 1).any():
                img_ndarray = img_ndarray / 255.0

        return img_ndarray
    
    
    
    def __getitem__(self, idx):
        
        name = self.ids[idx]
        
        img_file = list(self.Img_Dir.glob(name + '*.png'))  
        mask_file = list(self.Mask_Dir.glob(name + '*.png'))
              
        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])
        
        img = self.preprocessing(self.mask_values, img, self.Img_Size, self.scale, is_mask=False)
        mask = self.preprocessing(self.mask_values, mask, self.Img_Size, self.scale, is_mask=True)   
        
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.ids)