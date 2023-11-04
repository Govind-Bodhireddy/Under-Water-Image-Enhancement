import torch
import torch.nn as nn
class UWCNN(nn.Module):
    def __init__(self):
        super(UWCNN,self).__init__()
        self.conv_in=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv=nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv1=nn.Conv2d(in_channels=51,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=102,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv_out=nn.Conv2d(in_channels=153,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
    def forward(self,x):
        self.out_1=self.relu(self.conv_in(x))
        self.out_2=self.relu(self.conv(self.out_1))
        self.out_3=self.relu(self.conv(self.out_2))
        self.E_1=torch.cat((self.out_1,self.out_2,self.out_3,x),dim=1)
        self.out_4=self.relu(self.conv1(self.E_1))
        self.out_5=self.relu(self.conv(self.out_4))
        self.out_6=self.relu(self.conv(self.out_5))
        self.E_2=torch.cat((self.out_4,self.out_5,self.out_6,x,self.E_1),dim=1)
        self.out_7=self.relu(self.conv2(self.E_2))
        self.out_8=self.relu(self.conv(self.out_7))
        self.out_9=self.relu(self.conv(self.out_8))
        self.E_3=torch.cat((self.out_7,self.out_8,self.out_9,x,self.E_2),dim=1)
        self.out=self.conv_out(self.E_3)
        return self.out+x