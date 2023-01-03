from typing import List
import  typing
import os
import yaml
import argparse
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from dataset import SegDataset
from metrics import Measurement
from dice_loss import DiceLoss

from warmup_scheduler import GradualWarmupScheduler


class Trainer():
    def __init__(self, opt, cfg, model):
        print(opt)
        print(cfg)
        self.model = model
        self.start_epoch = 0
        self.num_epochs = cfg['NUM_EPOCHS']
        self.device = cfg['GPU']
        self.num_classes = cfg['NUM_CLASSES']
        # data load
        train_dataset = SegDataset(os.path.join(cfg['DATA_DIR'], 'train'), resize=cfg['RESIZE'], targetresize=True)
        val_dataset = SegDataset(os.path.join(cfg['DATA_DIR'], 'val'), resize=cfg['RESIZE'], targetresize=True)
        self.trainloader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
        self.valloader = DataLoader(val_dataset, 1, shuffle=False)
        
        # optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['OPTIM']['LR_INIT']), betas=(0.9, 0.999))
        # loss function
        self.loss = DiceLoss(num_classes=cfg['NUM_CLASSES'])
        # lr scheduler
        warmup_epochs=3
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs-warmup_epochs, eta_min=1e-7, verbose=True)
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler)
        self.lr_scheduler.step()
        
        self.measurement = Measurement(self.num_classes)
        # if resume
        if cfg['LOAD_WEIGHTS'] != '':
            print('############# resume training #############')
            self.resume = True
            self.device_setting(self.device)
            self.model = self.model.to(self.device)
            self.start_epoch, optimizer_statedict, scheduler_statedict, self.best_miou = self.load_checkpoint(cfg['LOAD_WEIGHTS'])
            self.optimizer.load_state_dict(optimizer_statedict)
            self.lr_scheduler.load_state_dict(scheduler_statedict)
            
            self.save_dir = os.path.split(os.path.split(cfg['LOAD_WEIGHTS'])[0])[0]
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
        else:    
            # save path
            self.resume = False
            os.makedirs(cfg['SAVE_DIR'], exist_ok=True)
            self.save_dir = os.path.join(cfg['SAVE_DIR'], f'{self.model.__class__.__name__}-ep{self.num_epochs}-'+str(len(os.listdir(cfg['SAVE_DIR']))))
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
            os.makedirs(self.ckpoint_path)
            

    def train(self, opt):
        ###debug###
        # torch.autograd.set_detect_anomaly(True)
        ############
        if not self.resume: 
            self.device_setting(self.device)
            self.model = self.model.to(self.device)
            self.best_miou = 0
        if opt.save_img:
            os.makedirs(os.path.join(self.save_dir, 'imgs'), exist_ok=True)
        if opt.save_txt:
            self.f = open(os.path.join(self.save_dir, 'result.txt'), 'a')
        if opt.save_graph or opt.save_csv : loss_list = []
        if opt.save_csv: 
            miou_list, lr_list = [], []
            self.val_miou_list = []
        
        self.best_miou_epoch = 0
        print('######### start training #########')
        for epoch in range(self.start_epoch, self.num_epochs) :
            ep_start = time.time()
            epoch_loss = 0
            epoch_miou = 0
            iou_per_class = np.array([0]*self.num_classes, dtype=np.float64)
            
            self.model.train()
            trainloader_len = len(self.trainloader)
            self.start_timer()
            for i, data in enumerate(tqdm(self.trainloader), 0):
                input_img, target_img = data[:2]
                label_img = self.mask_labeling(target_img, self.num_classes)
                input_img, label_img = input_img.to(self.device), label_img.to(self.device)
                
                # gradient initialization
                self.optimizer.zero_grad()
                # predict
                pred = self.model(input_img)
                # loss
                loss_output = self.loss(pred, label_img)
                loss_output.backward()
                # update
                self.optimizer.step()
                
                pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
                epoch_loss += loss_output.item()
                _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
                epoch_miou += ep_miou
                iou_per_class += ious
            epoch_loss /= trainloader_len
            epoch_miou /= trainloader_len
            epoch_ious = np.round((iou_per_class / trainloader_len), 5).tolist()
            
            if opt.save_graph:
                loss_list += [epoch_loss]
            if opt.save_csv:
                if not opt.save_graph: loss_list += [epoch_loss]
                miou_list += [epoch_miou]
                lr_list += [self.optimizer.param_groups[0]['lr']]
                
            traintxt = f"[epoch {epoch} Loss: {epoch_loss:.4f}, LearningRate :{self.optimizer.param_groups[0]['lr']:.6f}, trainmIOU: {epoch_miou}, train IOU per class:{epoch_ious}, time: {(time.time()-ep_start):.4f} sec \n" 
                
            print(traintxt)
            if opt.save_txt:
                self.f.write(traintxt)
            # save model
            self.save_checkpoint(epoch, 'model_last.pth')
        
            # validation
            self.val_test(epoch, opt)
            # lr scheduler update
            self.lr_scheduler.step()
        
        if opt.save_graph:
            self.save_lossgraph(loss_list)
        if opt.save_csv:
            self.save_csv('train', [loss_list, lr_list, miou_list], 'training.csv')
            self.save_csv('val', [self.val_miou_list], 'validation.csv')
            
        print("----- train finish -----")
        self.end_timer_and_print()
            
                
    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:'+device)
        else: 
            self.device = torch.device('cpu')  
    
    def val_test(self, epoch, opt):
        self.model.eval()
        val_miou = 0
        iou_per_class = np.array([0]*self.num_classes, dtype=np.float64)
        for i, data in enumerate(tqdm(self.valloader), 0):
            filename = data[-1]
            input_img, target_img = data[:2]
            label_img = self.mask_labeling(target_img, self.num_classes)
            input_img, label_img = input_img.to(self.device), label_img.to(self.device)
            with torch.no_grad():
                pred = self.model(input_img)
            pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
            _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
            val_miou += ep_miou
            iou_per_class += ious
            if opt.save_img:
                self.save_result_img(input_img.detach().cpu().numpy(), \
                    target_img.detach().cpu().numpy(), pred_numpy, filename, os.path.join(self.save_dir, 'imgs'))
                
        val_miou = val_miou / len(self.valloader)
        val_ious = np.round((iou_per_class / len(self.valloader)), 5).tolist()
        if val_miou >= self.best_miou:
            self.best_miou = val_miou
            self.best_miou_epoch = epoch
            self.save_checkpoint(epoch, 'best_miou.pth')
            
        valtxt = f"[val][epoch {epoch} mIOU: {val_miou:.4f}, IOU per class:{val_ious}---best mIOU:{self.best_miou}, best mIOU epoch: {self.best_miou_epoch}]\n"
        print(valtxt)
        # best miou model save
        if opt.save_txt:
            self.f.write(valtxt)
        if opt.save_csv:
            self.val_miou_list += [val_miou]
            
    def save_checkpoint(self, epoch, filename):
        filename = os.path.join(self.ckpoint_path, filename)
        torch.save({'network':self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer':self.optimizer.state_dict(),
                    'scheduler':self.lr_scheduler.state_dict(),
                    'best_miou':self.best_miou,},
                    filename)

    def load_checkpoint(self, weights_path, istrain=True):
        chkpoint = torch.load(weights_path)
        self.model.load_state_dict(chkpoint['network'])
        if istrain:
            return chkpoint['epoch'], chkpoint['optimizer'], chkpoint['scheduler'], chkpoint['best_miou']
        
    def mask_labeling(self, y_batch:torch.Tensor, num_classes:int):
        label_pixels = list(torch.unique(y_batch, sorted=True))
        assert len(label_pixels) <= num_classes, 'too many label pixels'
        if len(label_pixels) < num_classes:
            print('label pixels error')
            label_pixels = [0, 128, 255]
        
        for i, px in enumerate(label_pixels):
            y_batch = torch.where(y_batch==px, i, y_batch)

        return y_batch

    def pred_to_colormap(self, pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
        pred_label = np.argmax(pred, axis=1) # (N, H, W)
        show_pred = colormap[pred_label] # (N, H, W, 3)
        return show_pred

    def save_result_img(self, input:np.ndarray, target:np.ndarray, pred:np.ndarray, filename, save_dir):
        N = input.shape[0]
        show_pred = self.pred_to_colormap(pred)
        for i in range(N):
            input_img = np.transpose(input[i], (1, 2, 0)) # (H, W, 3)
            target_img = np.transpose(np.array([target[i]/255]*3), (1, 2, 0)) # (3, H, W) -> (H, W, 3)
            pred_img = show_pred[i] #(H, W, 3)
            cat_img = np.concatenate((input_img, target_img, pred_img), axis=1) # (H, 3W, 3)
            plt.imsave(os.path.join(save_dir, filename[i]), cat_img)
    
    def save_lossgraph(self, loss:list):
        # the graph for Loss
        plt.figure(figsize=(10,5))
        plt.title("Loss")
        plt.plot(loss, label='dice loss')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend() # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph.png'))
    
    def save_csv(self, mode, value_list:List, filename):
        if mode=='train':
            df = pd.DataFrame({'loss':value_list[0],
                                'lr':value_list[1],
                                'miou':value_list[2]
                                })
        if mode=='val':
            df = pd.DataFrame({'val_miou':value_list[0]})
            
        df.to_csv(os.path.join(os.path.abspath(self.save_dir), filename), mode='a')
    
    def start_timer(self):
        '''before training processes'''
        global start_time
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
        start_time = time.time()
    
    def end_timer_and_print(self):
        torch.cuda.synchronize()
        end_time = time.time()
        print("Total execution time = {:.3f} sec".format(end_time - start_time))
        print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='segmentation model''s name for training')
    parser.add_argument('--config', type=str, default='./config/seg_train_config.yaml', help='yaml file that has segmentation train config information')
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--save_csv', type=bool, default=True, help='save training process as csv file')
    parser.add_argument('--save_graph', type=bool, default=True, help='save Loss graph with plt')
    
    opt = parser.parse_args()
    
    if opt.model == 'unet':
        model = models.Unet(3)

    if opt.model == 'deeplabv3plus':
        model = models.Resnet50_DeepLabv3Plus()
        
    if opt.model == 'segnet':
        model = models.SegNet(3, 512, 3)

    if opt.model == 'cgnet':
        model = models.CGNet(3, 3)
        
    with open(opt.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    data_dirs = ['/content/data/restored_cropweed/DeblurGANv2/rice_s_n_w', '/content/data/restored_cropweed/HINet/rice_s_n_w', '/content/data/restored_cropweed/MIMO-Unet/rice_s_n_w',
                 '/content/data/restored_cropweed/MPRNet/rice_s_n_w', '/content/data/restored_cropweed/NAFNet/rice_s_n_w']
    save_dirs = ['../drive/MyDrive/restored_segtrain/DeblurGANv2/rice_s_n_w', '../drive/MyDrive/restored_segtrain/HINet/rice_s_n_w', '../drive/MyDrive/restored_segtrain/MIMO-Unet/rice_s_n_w',
                 '../drive/MyDrive/restored_segtrain/MPRNet/rice_s_n_w', '../drive/MyDrive/restored_segtrain/NAFNet/rice_s_n_w']
    # data_dirs = ['/content/data/restored_cropweed/proposed_patch_v2/IJRR2017']
    # save_dirs = ['../drive/MyDrive/restored_segtrain/proposed_patch_v2/IJRR2017']
    # data_dirs = ['/content/data/restored_ablation/numgroup12/CWFID', '/content/data/restored_ablation/numgroup34/CWFID',
    #              '/content/data/restored_ablation/numgroup123/CWFID', '/content/data/restored_ablation/numgroup234/CWFID']
    # save_dirs = ['../drive/MyDrive/ablation_segtrain/numgroup12/CWFID', '../drive/MyDrive/ablation_segtrain/numgroup34/CWFID',
    #              '../drive/MyDrive/ablation_segtrain/numgroup123/CWFID', '../drive/MyDrive/ablation_segtrain/numgroup234/CWFID']
    # data_dirs = ['/content/data/restored_ablation/noalpha/CWFID', '/content/data/restored_ablation/nodeformable/CWFID']
    # save_dirs = ['/content/drive/MyDrive/ablation_segtrain/noalpha/CWFID', '/content/drive/MyDrive/ablation_segtrain/nodeformable/CWFID']
    
    for datadir, savedir in zip(data_dirs, save_dirs):
        cfg['DATA_DIR'] = datadir
        cfg['SAVE_DIR'] = savedir
        trainer = Trainer(opt, cfg, model)
        trainer.train(opt)


    trainer = Trainer(opt, cfg, model)
    trainer.train(opt)