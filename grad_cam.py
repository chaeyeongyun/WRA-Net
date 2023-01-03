from glob import glob
import os
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import List
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm

def min_max_norm(input):
    if isinstance(input, np.ndarray):
        output = input - np.min(input) 
        output = output / (np.max(output) + 1e-7)
    if isinstance(input, torch.Tensor):
        output = input - torch.min(input) 
        output = output / (torch.max(output) + 1e-7)
    return output

def interpolate(input, size:tuple):
    ''' input (C, H, W) 
        output (C, H, W) '''
    return torch.squeeze(F.interpolate(torch.unsqueeze(input, dim=0), size=size), dim=0)

class GradCAM():
    def __init__(self, model:torch.nn.Module, target_layers:List[nn.Module], device:torch.device):
        self.model , self.target_layers = model, target_layers
        # if device == '-1':
        #     self.device = torch.device('cpu')
        # else:
        #     if torch.cuda.is_available(): 
        #         self.device = torch.device('cuda:'+device)
        #     else: raise Exception('this device is not available') 
        self.device = device
        self.gradients, self.activations = [], []
        for target_layer in target_layers:
            target_layer.register_forward_hook(self.save_activation_hook)
            target_layer.register_forward_hook(self.save_gradient_hook)
    
    def __call__(self, input_tensor:torch.Tensor, targets:List[nn.Module]):
        input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)

        self.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

        # activations_list = [x.detach().cpu().numpy() for x in self.activations]
        # grads_list = [x.detach().cpu().numpy() for x in self.gradients]
        activations_list = [x.detach() for x in self.activations]
        grads_list = [x.detach() for x in self.gradients]
        target_size = tuple(input_tensor.shape[-2:]) # (h, w)
        target_size = target_size[::-1] # (w, h)
        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = activations_list[i]
            layer_grads = grads_list[i]
            # cam_weights = np.mean(layer_grads, axis=(2, 3))
            cam_weights = torch.mean(layer_grads, dim=(2, 3))
            weighted_activations = cam_weights[:, :, None, None] * layer_activations # add axis and multiply
            # cam = weighted_activations.sum(axis=1) # (1, h, w)
            cam = torch.sum(weighted_activations, dim=1) # (1, h, w)
            # cam = np.maximum(cam, 0) # ReLU
            cam = torch.maximum(cam, torch.zeros_like(cam)) # ReLU
            cam = min_max_norm(cam)
            # cam = np.transpose(cam, (1, 2, 0)) # h, w, 1
            # cam = np.expand_dims(cv2.resize(cam, target_size), 0) # h, w -> 1, h, w
            cam = interpolate(cam, size=target_size)
            cam_per_target_layer += [cam[:, None, :]]
        # cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=0)
        cam_per_target_layer = torch.cat(cam_per_target_layer, dim=0)
        # cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        cam_per_target_layer = torch.maximum(cam_per_target_layer, torch.zeros_like(cam_per_target_layer))
        # final_cam = np.mean(cam_per_target_layer, axis=1)
        final_cam = torch.mean(cam_per_target_layer, dim=1)
        final_cam = min_max_norm(final_cam)
        # final_cam = cv2.resize(np.transpose(final_cam, (1, 2, 0)), target_size)
        final_cam = interpolate(final_cam, size=target_size)
        final_cam = torch.squeeze(final_cam)
        final_cam = final_cam.cpu().numpy()
        return final_cam
        
        
    def save_activation_hook(self, m, x, y):
        activation = y
        self.activations += [activation.cpu().detach()]
    
    def save_gradient_hook(self, m, x, y):
       assert hasattr(y, "requires_grad"), 'y must have ''requires_grad'' attribute'
       
       def _store_grad_hook(grad):
           self.gradients = [grad.cpu().detach()] + self.gradients
       
       y.register_hook(_store_grad_hook)
    
    

class GradCAMTarget:
    def __init__(self, class_num, mask:torch.Tensor):
        self.class_num = class_num
        self.mask = mask
        
    def __call__(self, model_output):
        self.mask = self.mask.to(model_output.device)
        return (model_output[self.class_num, :, :] * self.mask).sum()

def main(class_num, img_dir, model, model_weights_path, target_layers:List, device, save_path, ):
    if device == -1:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available(): 
            device = torch.device('cuda:'+device)
        else: raise Exception('this device is not available')
    model.load_state_dict(torch.load(model_weights_path)['network'])
    #### error model이 갑자기 unet객체가 아니게됨 띠용? 어디가 문제인지 확인하기
    model = model.to(device)
    print('model weights are loaded')
    def loop(class_num, imgs):
        for img in tqdm(imgs):
            filename = os.path.split(img)[-1]
            img = Image.open(img).convert('RGB')
            (orgw, orgh) = img.size
            img = np.array(img.resize((512, 512), resample=Image.BILINEAR))
            input_tensor = TF.to_tensor(img)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            mask =  make_mask(output, class_num)
            targets = [GradCAMTarget(class_num=class_num, mask=mask)]
            cam = GradCAM(model=model, target_layers=target_layers, device=device)
            gray_cam = cam(input_tensor=input_tensor, targets=targets)[0:]
            result_img = cam_on_img(img, gray_cam, img_weight=0.4, device=device)
            filename, extend = os.path.splitext(filename)[:]
            plt.imsave(os.path.join(save_path, filename+f'_cls{class_num}'+extend), cv2.resize(result_img, (orgw//2, orgh//2)))
    imgs = glob(os.path.join(img_dir, '*.png'))
    if isinstance(class_num, list):
        for i in class_num:
            loop(i, imgs)
    else:
        loop(class_num, imgs)
    
    

def make_mask(output, class_num):
    normalized_mask = torch.softmax(output, dim=1).cpu()
    normalized_mask = normalized_mask[0, :, :, :].argmax(axis=0).detach()
    mask_float =  normalized_mask==class_num
    mask_float = mask_float.type(torch.float32)
    return mask_float

def cam_on_img(img:np.ndarray, gray_cam:np.ndarray,img_weight:float, device:torch.device, colormap:int=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255*gray_cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255 # [0,1]
    heatmap = torch.from_numpy(heatmap).type(torch.float32)
    heatmap = heatmap / 255
    img = min_max_norm(img)
    img = torch.from_numpy(img)
    heatmap, img = heatmap.to(device), img.to(device)
    result = (1-img_weight) * heatmap + img_weight * img
    result = result / torch.max(result)
    return np.uint8(255*result.detach().cpu().numpy())
    
if __name__ == "__main__":
    model = models.Unet()
    target_layers = [model.expansive_path.conv4]

    main(class_num=[0, 1, 2], 
        img_dir='/content/data/cropweed_total/CWFID/seg/val/input', 
        model=model, 
        model_weights_path='/content/drive/MyDrive/segtrain/CWFID/Unet-ep50-0/ckpoints/best_miou.pth', 
        target_layers=target_layers, 
        device='0', 
        save_path='/content/drive/MyDrive/grad_cam_org/CWFID')
    
    