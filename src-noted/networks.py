#这个子程序的基本功能是设定神经网络的参数和结构
import torch
from torch import nn


## CNN
class ResidualBlock(nn.Module):  #残差块；ResNet的基本单元
    
    #进行卷积神经网络的初始化参数设置
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        
        super(ResidualBlock,self).__init__()
        
        #几层网咯如何排列
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:   #因为残差网络是要把跳过了卷积的和经过了卷积的一起运算，所以需要保证输入和输出一样大，不然还要维阿一次卷积调整尺寸
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
    #网络向前传播        
    def forward(self,x):
        
        residual = x     #residual是没有经过卷积运算的残差，直接进到下一轮
        
        x = self.cnn1(x)  #通过两层神经网络
        x = self.cnn2(x)
        
        x += self.shortcut(residual)  #按照加法运算
        
        x = nn.ReLU(True)(x)  #非线性化
        return x


 #本程序使用的完整的神经网络   
class ForkCNN(nn.Module):       
    
    def __init__(self, nFeatures, BatchSize, GPUs=1):
        
        self.features = nFeatures
        self.batch = BatchSize
        self.GPUs = GPUs
        
        super(ForkCNN, self).__init__()
        
        ### ResNet34，为什么直接用torch库内置的？
        self.resnet34 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(3,2),         #最大池化
            ResidualBlock(64,64),      #几个残差块的组合，PesidualBlock的两个参数依次是输入输出stride？
            ResidualBlock(64,64),      #前一个输出药等于后一个输入
            ResidualBlock(64,64,2),
            
            ResidualBlock(64,128),
            ResidualBlock(128,128),
            ResidualBlock(128,128),
            ResidualBlock(128,128,2),
            
            ResidualBlock(128,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256),
            ResidualBlock(256,256,2),
            
            ResidualBlock(256,512),
            ResidualBlock(512,512),
            ResidualBlock(512,512,2)
        )

        
        self.avgpool = nn.AvgPool2d(2)  #平均池化
        
        #cnn各层
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64,32,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32,16,kernel_size=3,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        
        ### Fully-connected layers全连接层
        #线性连接 Y=WX+b W为需要被训练的特征矩阵
        #经过三次操作最终得到特征矩阵
        self.fully_connected_layer = nn.Sequential(
            nn.Linear(1296, 512), # 1296 may vary with different image size
            nn.Linear(512, 128),
            nn.Linear(128, self.features),
        )

    
    #前向传播规则，（残差网络→池化）or卷积神经网络→flatten→concatnation→全连接层
    def forward(self, x, y):
        
        #这是两条平行路径？ 参考train程序，似乎前一个是针对星系图像，后一个是PSF
        x = self.resnet34(x)
        x = self.avgpool(x)
        
        #y = self.resnet18(y)
        y = self.cnn_layers(y)
        #y = self.avgpool(y)
        
        # Flatten
        x = x.view(int(self.batch/self.GPUs),-1)  #view对参数进行reshape，前一个参数设定转换后有几行，后一个参数指不提前指定转换前有几列
        y = y.view(int(self.batch/self.GPUs),-1)
        # print(x.size())
        # print(y.size())
        
        # Concatenation  #typo了xs
        z = torch.cat((x, y), -1)   #把两个Tensor连在一起  #猫猫函数doge
        #z = z.view(-1)
        #print(z.size())
        z = self.fully_connected_layer(z)
        
        return z

    
## NN calibration
class CaliNN(nn.Module):
    
    def __init__(self):
        super(CaliNN, self).__init__()
        
        self.main_net = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(4,5),
            #nn.ReLU(),
            nn.Linear(5,5),
            #nn.ReLU(),
            nn.Linear(5,5),
            #nn.ReLU(),
            nn.Linear(5,1),
        )

    
    def forward(self, x):
        
        x = self.main_net(x)
        return x
    
    


