import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def psnr(restored_batch:torch.Tensor, target_batch:torch.Tensor, pixel_max=1.0):
    '''
    Returns the average psnr value of each image in batch
    Args:
        restored_batch(ndarray) : restored image batch (deblurred) that have pixel values in [0 1], shape:(N, 3, H, W)
        target_batch(ndarray) : target image batch (original) that have pixel values in [0 1], shape:(N, 3, H, W)
    Returns: psnr 
    '''
    mse = torch.mean((target_batch - restored_batch)**2) # mean of flattened array
    if mse == 0 : return float('inf') 
    psnr = 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr

class PSNR(nn.Module):
    def __init__(self, pixel_max) :
        super().__init__()
        self.pixel_max = pixel_max
    def forward(self, restored, target):
        return psnr(restored, target, pixel_max=self.pixel_max)
    
#### SSIM
    
def ssim(restored_batch:torch.Tensor, target_batch:torch.Tensor, windowsize=11, sigma=1.5, reduction='mean'):
    '''
    Args:
        pred_batch(torch.Tensor) : restored image batch (deblurred) that have pixel values in [0 1], shape:(N, 3, H, W)
        target_batch(torch.Tensor) : target image batch (original) that have pixel values in [0 1], shape:(N, 3, H, W)
    Returns: ssim 
    '''
    assert restored_batch.shape == target_batch.shape, "two Tensors' shapes must be same"
    channel = restored_batch.shape[1]
    C1, C2 = 0.01**2, 0.03**2
    C3 = C2/2
    ## gaussian kernel
    gaussian_window = torch.Tensor(
        [math.exp(-(x - windowsize//2)**2/float(2*sigma**2)) for x in range(windowsize)] # Normal distribution pdf formula
        ) # torch.Size([11])
    gaussian_window = gaussian_window / gaussian_window.sum()
    window_1D = gaussian_window.unsqueeze(1) # torch.Size([11,1])
    window_2D = window_1D.mm(window_1D.t()) # matrix multiplication (11,1) x (11, 1) -> (11, 11)
    window_2D = window_2D.float().unsqueeze(0).unsqueeze(0) # (11,11) -> (1, 11, 11) -> (1, 1, 11, 11)
    window = window_2D.expand(channel, 1, windowsize, windowsize).contiguous()
    
    if restored_batch.is_cuda: window=window.to(restored_batch.get_device())
    window = window.type_as(restored_batch)
    
    # mu : luminance
    mu_x = F.conv2d(restored_batch, window, padding=windowsize//2, groups=channel) # (N, C, H, W)
    mu_y = F.conv2d(target_batch, window, padding=windowsize//2, groups=channel) # (N, C, H, W)
    mu_xy = mu_x * mu_y # (N, C, H, W)
    
    mu_x_sq, mu_y_sq = mu_x.pow(2), mu_y.pow(2) # (N, C, H, W)
    # sigma : contrast
    sigma_x_sq = F.conv2d(restored_batch*restored_batch, window, padding=windowsize//2, groups=channel) - mu_x_sq # (N, C, H, W)
    sigma_y_sq = F.conv2d(target_batch*target_batch, window, padding=windowsize//2, groups=channel) - mu_y_sq # (N, C, H, W)
    sigma_xy = F.conv2d(restored_batch*target_batch, window, padding=windowsize//2, groups=channel) - mu_xy # (N, C, H, W)
    
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2))/((mu_x_sq+mu_y_sq+C1)*(sigma_x_sq + sigma_y_sq + C2)) # (N, C, H, W)
    
    # reduction
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'sum':
        return ssim_map.sum()
    else:
        for i in range(len(restored_batch.shape)-1): ssim_map = ssim_map.mean(1)
        return ssim_map 
        
class SSIM(nn.Module):
    def __init__(self, windowsize=11, sigma=1.5, reduction='mean') :
        super(SSIM, self).__init__()
        self.windowsize = windowsize
        self.reduction = reduction
        self.sigma = sigma
    
    def forward(self, restored_batch:torch.Tensor, target_batch:torch.Tensor):
        return ssim(restored_batch, target_batch, windowsize=self.windowsize, sigma=self.sigma, reduction=self.reduction)
    
    
## segmentation metrics 
class Measurement:
    def __init__(self, num_classes:int, ignore_idx=None) :
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
    
    def _make_confusion_matrix(self, pred:np.ndarray, target:np.ndarray):
        """make confusion matrix

        Args:
            pred (numpy.ndarray): segmentation model's prediction score matrix
            target (numpy.ndarray): label
            num_classes (int): the number of classes
        """
        assert pred.shape[0] == target.shape[0], "pred and target ndarray's batchsize must have same value"
        N = pred.shape[0]
        # prediction score to label
        pred_label = pred.argmax(axis=1) # (N, H, W)
        
        pred_1d = np.reshape(pred_label, (N, -1)) # (N, HxW)
        target_1d = np.reshape(target, (N, -1)) # (N, HxW)
        # 3 * gt + pred = category
        cats = 3 * target_1d + pred_1d # (N, HxW)
        conf_mat = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.num_classes**2), axis=-1, arr=cats) # (N, 9)
        conf_mat = np.reshape(conf_mat, (N, self.num_classes, self.num_classes)) # (N, 3, 3)
        return conf_mat
    
    def accuracy(self, pred, target):
        '''
        Args:
            pred: (N, C, H, W), ndarray
            target : (N, H, W), ndarray
        Returns:
            the accuracy per pixel : acc(int)
        '''
        N = pred.shape[0]
        pred = pred.argmax(axis=1) # (N, H, W)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])) # (N, HxW)
        target = np.reshape(target, (target.shape[0], target.shape[1]*target.shape[2])) # (N, HxW)
        
        if self.ignore_idx != None:
             not_ignore_idxs = np.where(target != self.ignore_idx) # where target is not equal to ignore_idx
             pred = pred[not_ignore_idxs]
             target = target[not_ignore_idxs]
             
        return np.mean(np.sum(pred==target, axis=-1)/pred.shape[-1])
    
    def miou(self, conf_mat:np.ndarray, pred:np.ndarray, target:np.ndarray):
        iou_list = []
        sum_col = np.sum(conf_mat, -2) # (N, 3)
        sum_row = np.sum(conf_mat, -1) # (N, 3)
        for i in range(self.num_classes):
            batch_mean_iou = np.mean(conf_mat[:, i, i] / (sum_col[:, i]+sum_row[:, i]-conf_mat[:, i, i]+1e-8))
            iou_list += [batch_mean_iou]
        iou_ndarray = np.array(iou_list)
        miou = np.mean(iou_ndarray)
        return miou, iou_list
    
    def precision(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_col = np.sum(conf_mat, -2)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        precision_per_class = np.mean(np.array([conf_mat[:, i, i]/ sum_col[:, i] for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        # multi class에 대해 recall / precision을 구할 때에는 클래스 모두 합쳐 평균을 낸다.
        mprecision = np.mean(precision_per_class)
        return mprecision, precision_per_class

    def recall(self, conf_mat:np.ndarray):
        # confmat shape (N, self.num_classes, self.num_classes)
        sum_row = np.sum(conf_mat, -1)# (N, 3) -> 0으로 예측, 1로 예측 2로 예측 각각 sum
        recall_per_class = np.mean(np.array([conf_mat[:, i, i]/ sum_row[:, i] for i in range(self.num_classes)]), axis=-1) # list안에 (N, )가 클래스개수만큼.-> (3, N) -> 평균->(3,)
        mrecall = np.mean(recall_per_class)
        return mrecall, recall_per_class
    
    def f1score(self, recall, precision):
        return 2*recall*precision/(recall + precision)
    
    def measure(self, pred:np.ndarray, target:np.ndarray):
        conf_mat = self._make_confusion_matrix(pred, target)
        acc = self.accuracy(pred, target)
        miou, iou_list = self.miou(conf_mat, pred, target)
        precision, _ = self.precision(conf_mat)
        recall, _ = self.recall(conf_mat)
        f1score = self.f1score(recall, precision)
        return acc, miou, iou_list, precision, recall, f1score
        
    __call__ = measure