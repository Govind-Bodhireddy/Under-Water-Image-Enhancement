import torch
from skimage.color import rgb2hsv,hsv2rgb
import numpy as np
def pp(tensor):
    img=tensor.squeeze().detach().cpu().permute(2,1,0).numpy()
    hsv_image=rgb2hsv(img)
    hsv_image[:,:,1]=(hsv_image[:,:,1]-np.min(hsv_image[:,:,1]))/(np.max(hsv_image[:,:,1])-np.min(hsv_image[:,:,1]))
    rgb_image=hsv2rgb(hsv_image)
    return rgb_image
'''test_image=pp(torch.randn(1,3,310,230))
print(test_image.shape)'''
