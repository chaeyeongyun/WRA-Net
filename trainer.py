import os
import yaml
from typing import List
import argparse
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import models
from dataset import CropDataset, FullDataset
from metrics import psnr, ssim, SSIM
from warmup_scheduler import GradualWarmupScheduler


class Trainer():
    def __init__(self, opt, model):
        with open(opt.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print(opt)
        print(cfg)
        self.cfg=cfg
        self.device = cfg['GPU']
        self.model = model
        self.start_epoch = 0
        self.num_epochs = cfg['NUM_EPOCHS']
        self.half = opt.half
        print(self.half)
        
        train_dataset = CropDataset(os.path.join(cfg['DATA_DIR'], 'train'), cfg['PATCHSIZE'])
        val_dataset = CropDataset(os.path.join(cfg['DATA_DIR'], 'val'), cfg['PATCHSIZE'])
        
        self.trainloader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True)
        self.valloader = DataLoader(val_dataset, 1, shuffle=False)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['OPTIM']['LR_INIT']), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)


        self.ssim_loss = SSIM()
        self.l1_loss = nn.L1Loss()
        ## cosine annealing
        warmup_epochs = 3
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs*7-warmup_epochs, eta_min=1e-7, verbose=True)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_dataset)*(self.num_epochs)//1000-warmup_epochs, eta_min=1e-7, verbose=True)
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler)
        self.lr_scheduler.step()
        
        
        # if resume
        
        if cfg['LOAD_WEIGHTS'] != '':
            print('############# resume training #############')
            self.resume = True
            self.device_setting(self.device)
            self.model = self.model.to(self.device)
            self.start_epoch, optimizer_statedict, scheduler_statedict, self.best_psnr, self.best_ssim = self.load_checkpoint(cfg['LOAD_WEIGHTS'])
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
        if opt.save_img:
            os.makedirs(os.path.join(self.save_dir, 'imgs'), exist_ok=True)
        if opt.save_txt:
            self.f = open(os.path.join(self.save_dir, 'result.txt'), 'a')
        if opt.save_graph or opt.save_csv : loss_list = []
        if opt.save_csv: 
            psnr_list, ssim_list, lr_list = [], [], []
            self.val_loss_list, self.val_ssim_list, self.val_psnr_list = [], [], []
        
        
        print('######### start training #########')
        self.best_psnr, self.best_ssim = 0, 0
        self.best_psnr_epoch, self.best_ssim_epoch = 0, 0
        trainloader_len = len(self.trainloader)
        step = 0
        scaler = torch.cuda.amp.GradScaler(enabled=self.half)
        self.start_timer()
        epoch_loss = 0
        epoch_psnr, epoch_ssim = 0, 0
        for epoch in range(self.start_epoch, self.num_epochs) :
            ep_start = time.time()
            

            self.model.train()
            for i, data in enumerate(tqdm(self.trainloader), 0):
                step += 1
                input_img, target_img = data[:2]
                input_img, target_img = input_img.to(self.device), target_img.to(self.device)
                
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.half):
                    restored = self.model(input_img)
                    # loss_output = self.mse_loss(restored, target_img) + self.content_loss(restored, target_img)
                    alpha=0.84
                    loss_output = alpha*(1-self.ssim_loss(restored, target_img)) + (1-alpha)*self.l1_loss(restored, target_img)
                
                epoch_loss += loss_output.item()
                
                scaler.scale(loss_output).backward()
                # update parameters
                # self.optimizer.step()
                scaler.step(self.optimizer) # gradients가 infs 나 NaNs를 포함하지 않는다면 optimizer.step()이 호출된다.
                scaler.update()
                #metric
                restored_cpu, target_cpu = restored.detach().cpu(), target_img.detach().cpu()
                restored_cpu, target_cpu = list(map(lambda x: (x+1.0)/2.0, [restored_cpu, target_cpu]))[:]
                epoch_psnr += psnr(restored_cpu, target_cpu).item()
                epoch_ssim += ssim(restored_cpu, target_cpu).item()
                if step % 1000 == 0:
                     # train loss
                    epoch_loss /= 1000
                    if opt.save_graph:
                        loss_list += [epoch_loss]
                    #train metrics
                    epoch_psnr /= 1000
                    epoch_ssim /= 1000
                    if opt.save_csv:
                        if not opt.save_graph: loss_list += [epoch_loss]
                        psnr_list += [epoch_psnr]
                        ssim_list += [epoch_ssim]
                        lr_list += [self.optimizer.param_groups[0]['lr']]
                    
                    self.val_test(epoch, opt)
                    self.lr_scheduler.step()
                    torch.cuda.empty_cache()
                    traintxt = f"[epoch {epoch} Loss: {epoch_loss:.4f}, LearningRate :{self.optimizer.param_groups[0]['lr']:.6f}, trainPSNR: {epoch_psnr:.4f}, trainSSIM: {epoch_ssim:.4f}], time: {(time.time()-ep_start):.4f} sec \n" 
                    # % (epoch, epoch_loss, self.lr_scheduler.get_last_lr()[0], epoch_psnr, epoch_ssim, time.time()-ep_start)
                    print(traintxt)
                    if opt.save_txt:
                        self.f.write(traintxt)
                    epoch_loss = 0
                    epoch_psnr, epoch_ssim = 0, 0
                    # save model
                    self.save_checkpoint(epoch, 'model_last.pth')
                    
    
        # save model
        self.save_checkpoint(epoch, 'model_last.pth')
        if opt.save_graph:
            self.save_lossgraph(loss_list)
        if opt.save_csv:
            self.save_csv('train', [loss_list, lr_list, psnr_list, ssim_list], 'training.csv')
            self.save_csv('val', [self.val_loss_list, self.val_psnr_list, self.val_ssim_list], 'validation.csv')
            
        print("----- train finish -----")
        self.end_timer_and_print()
        
    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:'+device)
        else: 
            self.device = torch.device('cpu')       
    
    def val_test(self, epoch, opt):
        self.model.eval()
            
        val_loss, val_psnr, val_ssim = 0, 0, 0
        for input_img, target_img, filename in tqdm(self.valloader):
            input_img, target_img = input_img.to(self.device), target_img.to(self.device)
            
            with torch.no_grad():
                restored = self.model(input_img)
                alpha=0.84
                loss_output = alpha*(1-self.ssim_loss(restored, target_img)) + (1-alpha)*self.l1_loss(restored, target_img)
                val_loss += loss_output.item()
            # [-1, 1] to [0, 1]
            restored_cpu, target_cpu = restored.detach().cpu(), target_img.detach().cpu()
            restored_cpu, target_cpu = list(map(lambda x: (x+1.0)/2.0, [restored_cpu, target_cpu]))[:]
            
            val_psnr += psnr(restored_cpu, target_cpu).item()
            val_ssim += ssim(restored_cpu, target_cpu).item()
                
            if opt.save_img:
                self.save_img([target_cpu.numpy(), ((input_img.detach().cpu().numpy())+1.0)/2.0, restored_cpu.numpy()], filename)
        
        val_loss /= len(self.valloader)
        val_psnr /= len(self.valloader)
        val_ssim /= len(self.valloader)
        
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            self.best_psnr_epoch = epoch
            self.save_checkpoint(epoch, 'best_psnr.pth')
        if val_ssim > self.best_ssim:
            self.best_ssim = val_ssim
            self.best_ssim_epoch = epoch
            self.save_checkpoint(epoch, 'best_ssim.pth')

        valtxt = f"[val][epoch {epoch} PSNR: {val_psnr:.4f} SSIM: {val_ssim:.4f}--- best_psnr_epoch {self.best_psnr_epoch} Best_PSNR {self.best_psnr:.4f} / best SSIM epoch {self.best_ssim_epoch} Best_SSIM {self.best_ssim:.4f}]\n"
        print(valtxt)
        if opt.save_txt:
            self.f.write(valtxt)
        if opt.save_csv:
            self.val_loss_list += [val_loss]
            self.val_psnr_list += [val_psnr]
            self.val_ssim_list += [val_ssim]
        

    def save_checkpoint(self, epoch, filename):
        filename = os.path.join(self.ckpoint_path, filename)
        torch.save({'network':self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer':self.optimizer.state_dict(),
                    'scheduler':self.lr_scheduler.state_dict(),
                    'best_psnr':self.best_psnr,
                    'best_ssim':self.best_ssim},
                    filename)

    def load_checkpoint(self, weights_path, istrain=True):
        chkpoint = torch.load(weights_path)
        self.model.load_state_dict(chkpoint['network'])
        if istrain:
            return chkpoint['epoch'], chkpoint['optimizer'], chkpoint['scheduler'], chkpoint['best_psnr'], chkpoint['best_ssim']
                
    def save_img(self, img_list:List[np.ndarray], filename):
        """function to save image

        Args:
            img_list (List[np.ndarray]): [target, input, restored] images's list, target, input, resotred are np.ndarray and have (N, C, H, W) shape
        """
        concat_img = np.concatenate(img_list, axis=3) # (N, C, H, 3W)
        # transpose to (N, H, W, C) to save with plt
        concat_img = np.transpose(concat_img, (0, 2, 3, 1))
        for i, batch in enumerate(concat_img): # batch (N, H, W, C)
            plt.imsave(os.path.join(self.save_dir, 'imgs', filename[i]), batch)
        
    def save_lossgraph(self, loss:list):
        # the graph for Loss
        plt.figure(figsize=(10,5))
        plt.title("Loss")
        plt.plot(loss)
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend() # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph.png'))
    
    def save_csv(self, mode, value_list:List, filename):
        if mode=='train':
            df = pd.DataFrame({'loss':value_list[0],
                                'lr':value_list[1],
                                'psnr':value_list[2],
                                'ssim':value_list[3]})
        if mode=='val':
            df = pd.DataFrame({'val_loss':value_list[0], 
                               'val_psnr':value_list[1],
                                'val_ssim':value_list[2]})
            
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
        

        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/wranet_train_config.yaml', help='yaml file that has configuration for train')
    parser.add_argument('--half', type=bool, default=False, help='use mixed precision')
    parser.add_argument('--save_img', type=bool, default=False, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=False, help='save training process as txt file')
    parser.add_argument('--save_csv', type=bool, default=False, help='save training process as csv file')
    parser.add_argument('--save_graph', type=bool, default=False, help='save Loss graph with plt')
    
    opt = parser.parse_args()

    model = models.WRANet(3, 128)
    trainer = Trainer(opt, model)
    trainer.train(opt)