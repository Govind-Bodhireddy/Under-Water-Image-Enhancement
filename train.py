import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy
from PIL import Image
import model
import torch.nn as nn
import dataloader
from tqdm import tqdm


device='cuda' if torch.cuda.is_available() else 'cpu'
device_count=torch.cuda.device_count()
print(f"no of gpus={device_count}")
print(device)
model_0=model.UWCNN()
model_0=torch.nn.DataParallel(module=model_0,device_ids=[0,1])
model_0.cuda()
data_set=dataloader.dataload(train=True)
data=DataLoader(dataset=data_set,shuffle=True,batch_size=16)
print(len(data_set))
print(len(data))
loss_fn_1=nn.MSELoss()
optimizer=torch.optim.Adam(params=m),lr=0.0002,betas=(0.9,0.999))
epochs=40
training_loss=[]
for epoch in tqdm(range(epochs)):
    train_loss=0
    for b,(X,Y) in enumerate(data):
        model_0.train()
        X=X.cuda()
        Y=Y.cuda()
        y_pred=model_0.forward(X)
        loss=loss_fn_1(y_pred,Y)#+calc_ssim_loss(tensor_1=y_pred,tensor_2=Y)
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss/=len(data)
    training_loss.append(train_loss)
    print(f'training loss:{train_loss}')
with open('/home/govind/code_1/trn_losses/trn_loss_type__9_.pkl','wb') as f:
    pickle.dump(training_loss,f) 
torch.save(model_0.state_dict(),'/home/govind/md/t1')
