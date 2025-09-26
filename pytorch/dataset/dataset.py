import torch
from PIL import Image
import torch.utils.data as dataset
import os

#父类负责接口规范(规定子类必须包含的魔法方法)和兼容性，子类负责具体实现和数据内容
class MyData(dataset.Dataset):
    
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir   #定义根目录
        self.label_dir = label_dir #定义子目录(标签名)
        self.path = os.path.join(self.root_dir,self.label_dir) #拼接路径
        self.pathlist = os.listdir(self.path) #获取路径下的所有文件名，并形成列表
        
    def __getitem__(self, idx):
        image_name = self.pathlist[idx] #获取索引对应的文件名
        image_path = os.path.join(self.root_dir,self.label_dir,image_name) #拼接文件名路径
        image = Image.open(image_path) #读取图片地址，生成图片对象
        label = self.label_dir
        return image,label 

    def __len__(self):
        return len(self.pathlist) #返回文件名列表长度

root_dir = r"pytorch\train" #根目录
ants_label_dir = "ants_image" #子目录
bees_label_dir = "bees_image" #子目录
ants_dataset = MyData(root_dir,ants_label_dir) #子目录下数据集
bees_dataset = MyData(root_dir,bees_label_dir) #子目录下数据集

image,label = bees_dataset[6] #根据魔法方法的性质，输出__getitem__函数的返回值
print(image.show())

