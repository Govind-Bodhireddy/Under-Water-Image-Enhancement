import torch
import torch.nn.functional as F
import os 
from PIL import Image
import numpy as np
def ssim_loss(img1,img2,c1=0.02,c2=0.03,window_size=13):
    mu_1=F.avg_pool2d(img1,window_size,stride=1,padding=6)
    mu_2=F.avg_pool2d(img2,window_size,stride=1,padding=6)
    var_1=F.avg_pool2d(img1**2,window_size,stride=1,padding=6)-mu_1**2
    var_2=F.avg_pool2d(img2**2,window_size,stride=1,padding=6)-mu_2**2
    corr_12=F.avg_pool2d(img1*img2,window_size,stride=1,padding=6)-mu_1*mu_2
    ssim=((2*mu_1*mu_2+c1)*(2*corr_12+c2))/((mu_1**2+mu_2**2+c1)*(var_1+var_2+c2))
    ssim=torch.mean(ssim)
    return ssim.item()




def calc_ssim_loss(tensor_1,tensor_2):
    loss=0
    tensor_1=tensor_1.detach()
    tensor_2=tensor_2.detach()
    b,c,m,n=tensor_1.shape
    gray_tensor_1=torch.mean(tensor_1,dim=1)
    gray_tensor_2=torch.mean(tensor_2,dim=1)
    for i in range(b):
        t_gray_tensor_1=gray_tensor_1[i,:,:]
        gt_gray_tensor_1=gray_tensor_1[i,:,:]
        loss+=1-ssim_loss(img1=t_gray_tensor_1[None,:],img2=gt_gray_tensor_1[None,:])
    loss=loss/b
    return loss




def calc_mse_psnr(tensor1,tensor2):
    r_channel=torch.mean(((tensor1[0,:,:]-tensor2[0,:,:])**2))
    g_channel=torch.mean((tensor1[1,:,:]-tensor2[1,:,:])**2)
    b_channel=torch.mean((tensor1[2,:,:]-tensor2[2,:,:])**2)
    image_mean=(r_channel+b_channel+g_channel)/3

    r_channel_max=torch.max(tensor1[0,:,:])
    g_channel_max=torch.max(tensor1[1,:,:])
    b_channel_max=torch.max(tensor1[2,:,:])
    max_pixel=torch.max(torch.tensor([r_channel_max,g_channel_max,b_channel_max]))
    psnr=20*torch.log10(max_pixel)-10*torch.log10(image_mean)
    return image_mean.item(),psnr.item()




'''img1=torch.tensor(np.array(Image.open('/home/govind/data/type_1/type1_data/underwater_typeI/1_deep_3.7409_hori_12.3276_.bmp'))).permute(2,1,0).float()
img2=torch.tensor(np.array(Image.open('/home/govind/data/type_1/type1_data/gt_typeI/1_deep_3.7409_hori_12.3276_.bmp'))).permute(2,1,0).float()
img1=torch.mean(img1,dim=0)
img2=torch.mean(img2,dim=0)
print(img1.shape)
print(torch.mean(ssim_loss(img1=img1[None,:],img2=img2[None,:])))'''
