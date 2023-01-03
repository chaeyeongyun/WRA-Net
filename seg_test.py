import os
import argparse
import yaml
import gc
from tqdm import tqdm
from typing import List
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision

import sys
import models
from dataset import SegDataset
from metrics import Measurement

def mask_labeling(y_batch:torch.Tensor, num_classes:int) -> torch.Tensor:
    label_pixels = list(torch.unique(y_batch, sorted=True))
    
    if len(label_pixels) != num_classes:
        print('label pixels error')
        label_pixels = [0, 128, 255]
    
    for i, px in enumerate(label_pixels):
        y_batch = torch.where(y_batch==px, i, y_batch)

    return y_batch

def pred_to_colormap(pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    show_pred = colormap[pred_label] # (N, H, W, 3)
    return show_pred

def save_result_img(input:np.ndarray, target:np.ndarray, pred:np.ndarray, filename, save_dir):
    N = input.shape[0]
    show_pred = pred_to_colormap(pred)
    for i in range(N):
        input_img = np.transpose(input[i], (1, 2, 0)) # (H, W, 3)
        target_img = np.transpose(np.array([target[i]/255]*3), (1, 2, 0)) # (3, H, W) -> (H, W, 3)
        pred_img = show_pred[i] #(H, W, 3)
        cat_img = np.concatenate((input_img, target_img, pred_img), axis=1) # (H, 3W, 3)
        plt.imsave(os.path.join(save_dir, filename[i]), cat_img)


def test(model, opt):
    # torch.cuda.empty_cache()
    # gc.collect()
    print(opt)
    
    test_data = SegDataset(opt.data_dir, resize=512, targetresize=False)
    testloader = DataLoader(test_data, 1, shuffle=False)
    device = torch.device('cuda:'+opt.gpu) if opt.gpu != '-1' else torch.device('cpu')
    is_rst = opt.data_dir.split('/')[-3]
    save_dir = os.path.join(opt.save_dir, f'{is_rst}-{model.__class__.__name__}' + str(len(os.listdir(opt.save_dir))))
    if opt.save_img or opt.save_txt: os.makedirs(save_dir)
    
    num_classes = opt.num_classes
    measurement = Measurement(num_classes)
    # model = load_checkpoint(model, opt.weights)
    print('load weights...')
    try:
        model.load_state_dict(torch.load(opt.weights)['network'])
    except:
        model.load_state_dict(torch.load(opt.weights))
    model = model.to(device)
    if opt.save_txt:
        f = open(os.path.join(save_dir, 'results.txt'), 'w')
        f.write(f"data_dir:{opt.data_dir}, weights:{opt.weights}, save_dir:{opt.save_dir}")
    if opt.save_img:
        os.mkdir(os.path.join(save_dir, 'imgs'))
    
    model.eval()
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*opt.num_classes, dtype=np.float64)
    for input_img, mask_img, filename in tqdm(testloader):
        input_img = input_img.to(device)
        
        mask_cpu = mask_labeling(mask_img.type(torch.long), opt.num_classes)
        with torch.no_grad():
            pred = model(input_img)
        pred = F.interpolate(pred, mask_img.shape[-2:], mode='bilinear')
        pred_cpu, mask_cpu = pred.detach().cpu().numpy(), mask_cpu.cpu().numpy()
        
        acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_cpu, mask_cpu) 
        
        test_acc += acc_pixel
        test_miou += batch_miou
        iou_per_class += iou_ndarray
        
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
            
        if opt.save_img:
            input_img = F.interpolate(input_img.detach().cpu(), mask_img.shape[-2:], mode='bilinear')
            save_result_img(input_img.numpy(), mask_img.detach().cpu().numpy(), pred.detach().cpu().numpy(), filename, os.path.join(save_dir, 'imgs'))
    
    # test finish
    test_acc = test_acc / len(testloader)
    test_miou = test_miou / len(testloader)
    test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
    test_precision /= len(testloader)
    test_recall /= len(testloader)
    test_f1score /= len(testloader)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.8f, Test miou: %.8f" % (opt.weights,  test_acc, test_miou)       
    result_txt += f"\niou per class {test_ious}"
    result_txt += f"\nprecision : {test_precision}, recall : {test_recall}, f1score : {test_f1score} "
    print(result_txt)
    if opt.save_txt:
        f.write(result_txt)
        f.close()
    # torch.cuda.empty_cache()
    # gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # /data/LiteWRARN_afterdown_restored/test
    parser.add_argument('--data_dir', type=str, default='/content/data/restored_cropweed/proposed_patch_v2/CWFID/train', help='directory that has data')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/restored_segtest/proposed_patch_v2/CWFID', help='directory for saving results')
    parser.add_argument('--weights', type=str, default='/content/drive/MyDrive/restored_segtrain/proposed_patch_v2/CWFID/Unet-ep150-1/ckpoints/best_miou.pth', help='weights file for test')
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--show_img', type=bool, default=False, help='show images')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number. -1 is cpu')
    ##############
    parser.add_argument('--model', type=str, default='unet', help='modelname')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    
    opt = parser.parse_args()
    # assert opt.model in ['unet', 'deeplabv3', 'WRANet'], 'opt.model is not available'
    
    if opt.model == 'unet':
        model = models.Unet()
   
    test(model, opt)
    