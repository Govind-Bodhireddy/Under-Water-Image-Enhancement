import torch
from loss import ssim_loss,calc_mse_psnr
import pandas as pd
import pickle
def Metrics_test(tensor_1,tensor_2):
    n_imgs,_,_,_=tensor_1.shape
    ssim=[]
    mse=[]
    psnr=[]
    for i in range(n_imgs):
        gt_image=tensor_1[i,:,:,:].squeeze()
        test_img=tensor_2[i,:,:,:].squeeze()
        ssim.append(ssim_loss(img1=gt_image,img2=test_img))
        temp_mse,temp_psnr=calc_mse_psnr(tensor1=gt_image,tensor2=test_img)
        mse.append(temp_mse)
        psnr.append(temp_psnr)
    return ssim,mse,psnr
'''ssim,mse,psnr=Metrics_test(torch.ones(16,3,310,230),torch.ones(16,3,310,230))
l=['SSIM','MSE','PSNR']
df=pd.DataFrame([ssim,mse,psnr])
df=df.transpose()
df.columns=l
print(df)
with open('/home/govind/code_1/test_metrics/type1.pkl','wb') as f:
    pickle.dump(df,f)
print(len(ssim))
print(len(mse))
print(len(psnr))
print(ssim)
print(mse)
print(psnr)'''
