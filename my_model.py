import torch
import torch.nn as nn
class myUWCNN(nn.Module):
    def __init__(self):
        super(myUWCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=3+64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=3+64+64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(in_channels=3+64+64+64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv5=nn.Conv2d(in_channels=3+64,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
    def forward(self,x):
        self.out1=self.relu(self.conv1(x))
        self.E_1=torch.cat((self.out1,x),dim=1)
        self.out2=self.relu(self.conv2(self.E_1))
        self.E_2=torch.cat((self.out2,self.out1,x),dim=1)
        self.out3=self.relu(self.conv3(self.E_2))
        self.E_3=torch.cat((self.out3,self.out2,self.out1,x),dim=1)
        self.out4=self.relu(self.conv4(self.E_3))
        self.E5=torch.cat((self.out4,x),dim=1)
        self.out5=self.conv5(self.E5)
        return self.out5+x
m=myUWCNN()
y=m(torch.randn(16,3,230,310))
print(y.shape)