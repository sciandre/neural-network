import numpy
import scipy
import neural_net
import matplotlib.pyplot

digital_recognition = neural_net.smile(inputnodes=784,hiddennodes=100,outputnodes=10,learningrate=0.3)

#打开数据文件
data_file = open('numpy\mnist_train_100.csv','r') #打开训练文件，只读
data_list = data_file.readlines()           #读取所有行，转化为字符串列表
data_file.close()                         #关闭文件

for j in range(6):
    for i in range(len(data_list)):
        #对数据进行处理，生成输入值
        str_list=data_list[i].split(',')                #将字符串分割，形成一个字符串列表
        inputs = [(float(x)/255.0*0.99+0.01) for x in str_list[1:]]     #将字符串列表转换为浮点数列表
        #生成目标值
        onodes = 10
        targets = numpy.zeros(onodes) + 0.01
        targets[int(str_list[0])] = 0.99
        data_train = digital_recognition.train(inputs_list = inputs,targets_list = targets)
        if int(str_list[0]) == int(numpy.argmax(data_train)):
            print(f"手写数字{str_list[0]}识别为{numpy.argmax(data_train)}的概率为{max(data_train)[0]*100:.2f}%,")

numpy.set_printoptions(threshold=numpy.inf, linewidth=numpy.inf)
who_matrix = open('output_matrix','w')
print(digital_recognition.who,file=who_matrix)
wih_matrix = open('hidden_matrix','w')
print(digital_recognition.wih,file=wih_matrix)
numpy.savetxt('wih_matrix.txt', digital_recognition.wih)

