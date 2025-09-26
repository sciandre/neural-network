import ai
import matplotlib.pyplot
import pandas
import torch
import time

model=ai.torch_ai()

#计时1
start_time=time.time()
#训练ai
mnist_train=ai.MnistDataset('python\pytorch\mnist_train_100.csv')
epochs=10
for i in range (epochs):
    print('training epoch',i+1,'of',epochs)
    for label,image_data_tensor,target_tensor in mnist_train:
        model.train(image_data_tensor,target_tensor)
        pass
    pass
#计时2
end_time = time.time()
print(f"训练耗时：{end_time - start_time:.2f} 秒")

model.plot_progress()

#单个测试
mnist_test=ai.MnistDataset('python\pytorch\mnist_test_10.csv')
record=1
mnist_test.plot_image(record)
image_data=mnist_test[record][1]
output_cpu=model.forward(image_data)
pandas.DataFrame(output_cpu.detach().numpy()).plot(kind='bar',legend=False,ylim=(0,1))
matplotlib.pyplot.show()

#测试整个测试集
score=0
items=10

for label,image_data_tensor,target_tensor in mnist_test:
    answer=model.forward(image_data_tensor).detach().numpy()
    if(answer.argmax() == label):
        score += 1
        pass
print(f'准确率为{100*score/items}%')
