import os
import argparse
import yaml
from tqdm import tqdm
from typing import List
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
import models
from dataset import PatchDataset
from metrics import psnr, ssim


def save_compare_img(img_list:List[np.ndarray], filename, save_dir):
    """function to save image

    Args:
        img_list (List[np.ndarray]): [target, input, restored] images's list, target, input, resotred are np.ndarray and have (N, C, H, W) shape
    """
    concat_img = np.concatenate(img_list, axis=3) # (N, C, H, 3W)
    # transpose to (N, H, W, C) to save with plt
    concat_img = np.transpose(concat_img, (0, 2, 3, 1))
    for i, batch in enumerate(concat_img): # batch (N, H, W, C)
        plt.imsave(os.path.join(save_dir, 'imgs', filename[i]), batch)

def save_restored_img(restored_img:np.ndarray, filename, save_dir):
    img = np.transpose(restored_img, (0, 2, 3, 1))
    for i, batch in enumerate(img):
        plt.imsave(os.path.join(save_dir, 'restored_imgs', filename[i]), batch)
    
def load_checkpoint(model, weights_path):
    chkpoint = torch.load(weights_path)
    print('It''s %d epoch weights' % (chkpoint['epoch']))
    print("### Loading weights ###")
    model.load_state_dict(chkpoint['network'])

def test(model, opt):
    print(opt)
    test_data = PatchDataset(opt.data_dir, opt.num_patch, inter=10, factor=4)
    testloader = DataLoader(test_data, 1, shuffle=False)
    device = torch.device('cuda:'+opt.gpu) if opt.gpu != '-1' else torch.device('cpu')
    save_dir = os.path.join(opt.save_dir, f'{model.__class__.__name__}' + str(len(os.listdir(opt.save_dir))))
    if opt.save_img or opt.save_txt: os.makedirs(save_dir)
    
    load_checkpoint(model, opt.weights)
    model = model.to(device)
    if opt.save_txt:
        f = open(os.path.join(save_dir, 'results.txt'), 'w')
        f.write(f"{opt.data_dir}, ") 
    if opt.save_img:
        os.mkdir(os.path.join(save_dir, 'imgs'))
        os.mkdir(os.path.join(save_dir, 'restored_imgs'))
    
    model.eval()
    test_psnr, test_ssim = 0, 0
    total_time = 0 # millisecond

    for ip_list, target_img, input_img, filename in tqdm(testloader):

        zeros = torch.zeros(target_img.shape).to(device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for patch in ip_list:
            coords = patch[1]
            patch = patch[0].to(device) # (N, C, H, W)
            orgh, orgw = patch.shape[2:] # length 4 list, 각 요소는 N개의 h0 값을 갖고있는 tensor, N개의 h1값을 갖고있는 tensor, ...
            factor = 4
            H,W = ((orgh+factor)//factor)*factor, ((orgw+factor)//factor)*factor
            padh = H-orgh if orgh%factor!=0 else 0
            padw = W-orgw if orgw%factor!=0 else 0
            patch = F.pad(patch, (0,padw,0,padh), 'reflect')
            with torch.no_grad():
                starter.record()
                restored = model(patch)[:, :, :orgh, :orgw].detach()
                zeros[:, :, coords[0][0]:coords[1][0], coords[2][0]:coords[3][0]] = \
                    torch.where(zeros[:, :, coords[0][0]:coords[1][0], coords[2][0]:coords[3][0]]==0, restored, (zeros[:, :, coords[0][0]:coords[1][0], coords[2][0]:coords[3][0]]+restored)/2)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(curr_time)
        total_time += curr_time

        # [-1, 1] to [0, 1]
        restored_cpu = zeros.detach().cpu()

        restored_cpu, target_img = list(map(lambda x: (x+1.0)/2.0, [restored_cpu, target_img]))[:]
        test_psnr += psnr(restored_cpu, target_img).item()
        test_ssim += ssim(restored_cpu, target_img).item()
            
        if opt.save_img:
            save_compare_img([target_img.numpy(), (input_img.numpy()+1.0)/2.0, restored_cpu.numpy()], filename, save_dir)
            save_restored_img(restored_cpu.numpy(), filename, save_dir)
            
    test_psnr /= len(testloader)
    test_ssim /= len(testloader)
    
    testtxt = f"PSNR: {test_psnr:.4f} SSIM: {test_ssim:.4f}\n"
    testtxt += f"mean time(ms) : {total_time/len(testloader)}"
    print(testtxt)
    if opt.save_txt:
        f.write(testtxt)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/cropweed_total/CWFID/MPR/test', help='directory that has data')
    parser.add_argument('--save_dir', type=str, default='../drive/MyDrive/test/CWFID', help='directory for saving results')
    parser.add_argument('--weights', type=str, default='/content/drive/MyDrive/train/CWFID/WRANet-ep20-0/ckpoints/best_psnr.pth', help='weights file for test')
    parser.add_argument('--save_img', type=bool, default=False, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=False, help='save training process as txt file')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number. -1 is cpu')
    ##############
    parser.add_argument('--num_patch', type=int, default=3, help='Number of patches to split the image (split by number of num_patch x num_patch)')
    parser.add_argument('--model', type=str, default='wra-net', help='modelname')
    
    opt = parser.parse_args()
   
    if opt.model == 'wra-net':
        model = models.WRANet(3)
 
    test(model, opt)
    
    