import numpy as np
import scipy

class smile:
    #初始化各层节点、学习因子、权重矩阵、激活函数
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #self使得类里面的其他函数都可以使用init初始化的参数
        self.inodes=inputnodes  #输入节点数
        self.hnodes=hiddennodes #隐藏层节点数
        self.onodes=outputnodes #输出层节点数
        self.lr=learningrate    #学习因子
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)) #隐藏层权重矩阵
        #np.random.normal()：np.随机.正态分布函数(均值，标准差，生成随机数组的形状)
        #pow()：求幂运算函数(底数,指数)
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5,),(self.onodes,self.hnodes)) #输出层权重矩阵
        self.activation_function=lambda x: scipy.special.expit(x) #激活函数
        #lambda x定义一个匿名函数self.activation_function,函数内部结构为scipy.special.expit(x)
        #若去掉lambda x则表示将scipy.special.expit(x)的计算结果赋予给变量self.activation_function
        #scipy.special.expit(x),sigmoid激活函数
        pass

    def train(self,inputs_list,targets_list):
        inputs=np.array(inputs_list,ndmin=2).T                  #将输入列表转化为输入列矩阵
        targets=np.array(targets_list,ndmin=2).T                #将目标列表转化为输入列矩阵
        #np.array()创建数组函数 ndim=2生成二维数组 .T对数组进行转置操作
        hidden_inputs=np.dot(self.wih,inputs)                   #隐藏层矩阵运算
        hidden_outputs=self.activation_function(hidden_inputs)  #隐藏层激活函数计算
        final_inputs=np.dot(self.who,hidden_outputs)            #输出层矩阵运算
        final_outputs=self.activation_function(final_inputs)    #输出层激活函数计算
        output_errors=targets-final_outputs                     #输出层误差
        hidden_errors=np.dot(self.who.T,output_errors)          #隐藏层误差
        #隐藏层误差=隐藏层矩阵的转置*输出层误差
        self.who+=self.lr*np.dot(output_errors*final_outputs*(1.0-final_outputs),np.transpose(hidden_outputs)) #调整后的输出层矩阵
        self.wih+=self.lr*np.dot(hidden_errors*hidden_outputs*(1.0-hidden_outputs),np.transpose(inputs))       #调整后的隐藏层矩阵
        return final_outputs
        pass

    def query(self,inputs_list):
        inputs=np.array(inputs_list,ndmin=2).T                  #将输入列表转化为输入列矩阵
        #np.array()创建数组函数 ndim=2生成二维数组 .T对数组进行转置操作
        hidden_inputs=np.dot(self.wih,inputs)                   #隐藏层矩阵运算
        hidden_outputs=self.activation_function(hidden_inputs)  #隐藏层激活函数计算
        final_inputs=np.dot(self.who,hidden_outputs)            #输出层矩阵运算
        final_outputs=self.activation_function(final_inputs)    #输出层激活函数计算
        return final_outputs
        pass
