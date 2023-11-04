import torch
import dataloader
import metrics
import model
import pandas as pd
import pickle
from torch.utils.data import DataLoader
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_test=model.UWCNN()
model_test.cuda()
model_type=input('Enter type of model for test: ')
if model_type=='model_1':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model_1'))
elif model_type=='model_1A':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model_1A'))
elif model_type=='model_1B':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model_1B'))
elif model_type=='model_II':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model_II'))
elif model_type=='model_III':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model_III'))
elif model_type=='model__1_':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model__1_'))
elif model_type=='model__3_':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model__3_'))
elif model_type=='model__5_':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model__5_'))
elif model_type=='model__7_':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model__7_'))
elif model_type=='mode;__9_':
    model_test.load_state_dict(torch.load('/home/govind/code_1/models/model__9_'))
else:
    Exception('Enter valid mode_type')
model_test=torch.nn.DataParallel(module=model_test,device_ids=[0,1])
model_test.eval()
data_set=dataloader.dataload(train=False)
data=DataLoader(data_set,batch_size=16)
SSIM=[]
MSE=[]
PSNR=[]
for b,(X,Y) in enumerate(data):
    #print(b)
    #print(X.shape)
    X=X.cuda()
    Y=Y.cuda()
    with torch.inference_mode():
        test_pred=model_test.forward(X)
        ssim,mse,psnr=metrics.Metrics_test(test_pred.detach().cpu(),Y.detach().cpu())
        SSIM=SSIM+ssim
        MSE=MSE+mse
        PSNR=PSNR+psnr
        print(len(SSIM))
        print(len(ssim))
l=['SSIM','MSE','PSNR']
df=pd.DataFrame([SSIM,MSE,PSNR])
df=df.transpose()
df.columns=l
with open('/home/govind/code_1/test_metrics/type__9_.pkl','wb') as f:
    pickle.dump(df,f)




