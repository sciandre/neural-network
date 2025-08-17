import torch
import pandas
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

#定义神经网络类 torch_ai，继承声明PyTorch 基类 nn.Module的参数管理、前向传播、模型保存等核心功能方法
class torch_ai(nn.Module):

    #初始化函数
    def __init__(self):
        super().__init__() #继承父类构造函数，提供连接功能函数和输入参数的API端口,初始化父类注册表，将参数放入注册表中管理

        #nn.Sequential()是用按顺序堆叠的方式构造神经网络的函数，输出分类任务中各类别的预测概率
        self.model=nn.Sequential(
            nn.Linear(784,200), #神经网络的线性层
            nn.Sigmoid(), #S激活函数
            nn.Linear(200,10), 
            nn.Sigmoid()
            
        )
        self.loss_fuction=nn.MSELoss() #均方差损失函数
        self.optimiser=torch.optim.SGD(self.parameters(),lr=0.01) #将SGD作为优化器负责根据损失函数计算的梯度更新链接权重
        self.counter=0 #初始化训练次数变量
        self.progress=[] #初始化记录损失值的列表
        pass

    #输出函数
    def forward(self,inputs):
        return self.model(inputs)

    #训练函数
    def train(self,inputs,targets):
        outputs=self.forward(inputs)
        loss=self .loss_fuction(outputs,targets)
        self.counter += 1
        if(self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if(self.counter % 1000 == 0):
            print('counter=',self.counter)
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    #损失图函数
    def plot_progress(self):
        df=pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0,1.0),figsize=(16,8),alpha=0.1,marker='.',grid=True,yticks=(0,0.25,0.5))
        plt.show()
        pass
    pass

#数据集类，用于处理输入数据
class MnistDataset(Dataset):
    def __init__(self,csv_file):
        self.data_df=pandas.read_csv(csv_file,header=None)
        pass

    def __len___(self):
        return len(self.data_df)

    #获取数据集里的标签张量，图像数据张量和目标张量
    def __getitem__(self,index):
        label=self.data_df.iloc[index,0]
        target=torch.zeros((10))
        target[label]=1.0
        image_values=torch.FloatTensor(self.data_df.iloc[index,1:].values)/255.0
        return label,image_values,target

    #输出分类任务预测概率的柱状图
    def plot_image(self,index):
        arr=self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title('label='+str(self.data_df.iloc[index,0]))
        plt.imshow(arr,interpolation='none',cmap='Blues')
        plt.show()
        pass
    pass