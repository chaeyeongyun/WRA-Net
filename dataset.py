import os
import glob
import numpy as np
import math
import random
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def change_pixel_range(x:torch.Tensor):
    """chanhe pixel range to [-1, 1] from [0, 1]

    Args:
        x (torch.Tensor): input torch Tensor

    Returns:
        y : torch Tensor that have value in range[ -1, 1 ]
    """
    y = 2*(x-0.5)
    return y


class CropDataset(Dataset):
    ''' for train '''
    def __init__(self, data_dir, cropsize, randomaug=True):
        input_images = sorted(os.listdir(os.path.join(data_dir, 'input'))) # ['rgb_00254_2.png', 'rgb_00094_2.png', 'rgb_00185.png', ... ]
        target_images = sorted(os.listdir(os.path.join(data_dir, 'target')))
        assert len(input_images) == len(target_images), "the number of input images and output images must be the same"
        
        self.input_path = [os.path.join(data_dir, 'input', x) for x in input_images]
        self.target_path = [os.path.join(data_dir, 'target', x) for x in target_images]
        
        self.cropsize = cropsize
        self.randomaug = randomaug
        
    def __len__(self):
        return len(self.input_path)
    
    def __getitem__(self, idx): 
        cropsize = self.cropsize
        assert self.input_path[idx].split('/')[-1] == self.target_path[idx].split('/')[-1], \
                    "input image and target image are not matched"
        input_img = Image.open(self.input_path[idx]).convert('RGB')
        target_img = Image.open(self.target_path[idx]).convert('RGB')
        filename = self.input_path[idx].split('/')[-1] 
        if input_img.size != target_img.size: 
            target_img = target_img.resize(input_img.size)
            
        assert all([x >= cropsize for x in input_img.size]), "cropsize must smaller than image size"
       

        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)
        
        if self.randomaug:
            # Data Augmentations
            aug = random.randint(0, 7)
            if aug==1:
                input_img = input_img.flip(1)
                target_img = target_img.flip(1)
            elif aug==2:
                input_img = input_img.flip(2)
                target_img = target_img.flip(2)
            elif aug==3:
                input_img = torch.rot90(input_img,dims=(1,2))
                target_img = torch.rot90(target_img,dims=(1,2))
            elif aug==4:
                input_img = torch.rot90(input_img,dims=(1,2), k=2)
                target_img = torch.rot90(target_img,dims=(1,2), k=2)
            elif aug==5:
                input_img = torch.rot90(input_img,dims=(1,2), k=-1)
                target_img = torch.rot90(target_img,dims=(1,2), k=-1)
            elif aug==6:
                input_img = torch.rot90(input_img.flip(1),dims=(1,2))
                target_img = torch.rot90(target_img.flip(1),dims=(1,2))
            elif aug==7:
                input_img = torch.rot90(input_img.flip(2),dims=(1,2))
                target_img = torch.rot90(target_img.flip(2),dims=(1,2))
        
        
        # crop patch
        random.seed(1)
        x1 = random.randint(0, input_img.size()[2]-cropsize)
        y1 = random.randint(0, input_img.size()[1]-cropsize)
        input_img = input_img[:, y1:y1+cropsize, x1:x1+cropsize]
        target_img = target_img[:, y1:y1+cropsize, x1:x1+cropsize]
        
        input_img, target_img = change_pixel_range(input_img), change_pixel_range(target_img) # Tensor that have values in range [-1, 1]
        return input_img, target_img, filename
        

        
class FullDataset(Dataset):
    ''' for test '''
    def __init__(self, data_dir, randomaug=False):
        input_images = sorted(os.listdir(os.path.join(data_dir, 'input'))) # ['rgb_00254_2.png', 'rgb_00094_2.png', 'rgb_00185.png', ... ]
        target_images = sorted(os.listdir(os.path.join(data_dir, 'target')))
        assert len(input_images) == len(target_images), "the number of input images and output images must be the same"
        
        self.input_path = [os.path.join(data_dir, 'input', x) for x in input_images]
        self.target_path = [os.path.join(data_dir, 'target', x) for x in target_images]
        
        self.randomaug = randomaug
        
    def __len__(self):
        return len(self.input_path)
    
    def __getitem__(self, idx):
        assert self.input_path[idx].split('/')[-1] == self.target_path[idx].split('/')[-1], \
                    "input image and target image are not matched"
                    
        filename = self.input_path[idx].split('/')[-1] 
        input_img = Image.open(self.input_path[idx]).convert('RGB')
        target_img = Image.open(self.target_path[idx]).convert('RGB')
         
        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)
        
        if self.randomaug:   
            # Data Augmentations
            aug = random.randint(0, 7)
            if aug==1:
                input_img = input_img.flip(1)
                target_img = target_img.flip(1)
            elif aug==2:
                input_img = input_img.flip(2)
                target_img = target_img.flip(2)
            elif aug==3:
                input_img = torch.rot90(input_img,dims=(1,2)) # 좌로 90도 회전
                target_img = torch.rot90(target_img,dims=(1,2))
            elif aug==4:
                input_img = torch.rot90(input_img,dims=(1,2), k=2) # k:횟수 (H, W)기준 2번 좌로 90도 회전이니까 180도 회전
                target_img = torch.rot90(target_img,dims=(1,2), k=2)
            elif aug==5:
                input_img = torch.rot90(input_img,dims=(1,2), k=-1) # k<0이면 우로 회전. 우로 90도 회전
                target_img = torch.rot90(target_img,dims=(1,2), k=-1)
            elif aug==6:
                input_img = torch.rot90(input_img.flip(1),dims=(1,2)) # dim 1(H) 기준 flip 하고, 좌로 90도 회전
                target_img = torch.rot90(target_img.flip(1),dims=(1,2))
            elif aug==7:
                input_img = torch.rot90(input_img.flip(2),dims=(1,2))  # dim 1(W) 기준 flip 하고, 좌로 90도 회전
                target_img = torch.rot90(target_img.flip(2),dims=(1,2))
        
        
        
        input_img, target_img = change_pixel_range(input_img), change_pixel_range(target_img) # Tensor that have values in range [-1, 1]
        return input_img, target_img, filename

class SegDataset(Dataset):
    '''for Unet'''
    def __init__(self, data_dir, resize=512, inputresize=True, targetresize=False, transform=None, target_transform=None):
        self.img_dir = os.path.join(data_dir, 'input')
        self.mask_dir = os.path.join(data_dir, 'target')
        self.resize = resize
        self.inputresize = inputresize
        self.targetresize = targetresize
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform    
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.inputresize: image = image.resize((self.resize, self.resize), resample=Image.BILINEAR) 
        image = TF.to_tensor(image)
        
        mask_path = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_path).convert('L') # size : (W, H), grayscale image
        if self.targetresize: mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)
        mask = np.array(mask) # (H, W)
        mask = torch.from_numpy(mask)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask, filename



class PatchDataset(Dataset):
    '''for inference at small memory environment'''
    '''겹치게 잘라서 겹치는 부분 평균내는 방법'''
    def __init__(self, data_dir, num_patch:int, inter:int, factor:int=None) :
        super().__init__()
        self.num_patch = num_patch
        self.inter = inter
        input_images = sorted(os.listdir(os.path.join(data_dir, 'input'))) # ['rgb_00254_2.png', 'rgb_00094_2.png', 'rgb_00185.png', ... ]
        target_images = sorted(os.listdir(os.path.join(data_dir, 'target')))
        assert len(input_images) == len(target_images), "the number of input images and output images must be the same"
        
        self.input_path = [os.path.join(data_dir, 'input', x) for x in input_images]
        self.target_path = [os.path.join(data_dir, 'target', x) for x in target_images]
        
        self.factor = factor
    
    def __len__(self):
        return len(self.input_path)
    
    def __getitem__(self, idx) :
        input_img = Image.open(self.input_path[idx]).convert('RGB')
        target_img = Image.open(self.target_path[idx]).convert('RGB')
        filename = os.path.split(self.input_path[idx])[-1] 
        
        input_img = TF.to_tensor(input_img)
        target_img = TF.to_tensor(target_img)
        input_img, target_img = change_pixel_range(input_img), change_pixel_range(target_img) # Tensor that have values in range [-1, 1]
        
        # patches of input
        ip_list = []
        h, w = input_img.shape[-2], input_img.shape[-1]        
        patch_h, patch_w = h // self.num_patch, w // self.num_patch
        if self.factor != None:
            patch_h = ((patch_h+self.factor)//self.factor)*self.factor
            patch_w = ((patch_w+self.factor)//self.factor)*self.factor
        coord_h = [(patch_h*i-self.inter if i>0 else patch_h*i, patch_h*(i+1)+self.inter) for i in range(self.num_patch-1)] + [(patch_h*(self.num_patch-1)-self.inter, h)]
        coord_w = [(patch_w*i-self.inter if i>0 else patch_w*i, patch_w*(i+1)+self.inter) for i in range(self.num_patch-1)] + [(patch_w*(self.num_patch-1)-self.inter, w)]
        for c_h in coord_h:
            for c_w in coord_w:
                ip_list += [(input_img[:, c_h[0]:c_h[1], c_w[0]:c_w[1]], (c_h[0], c_h[1], c_w[0], c_w[1]))] # (img, h좌료 튜플, w좌표 튜플) 가진 리스트

        return ip_list, target_img, input_img, filename

