import ai
import matplotlib.pyplot
import pandas

C=ai.torch_ai()

#训练ai
mnist_train=ai.MnistDataset('python\pytorch\mnist_train_100.csv')
epochs=10
for i in range (epochs):
    print('training epoch',i+1,'of',epochs)
    for label,image_data_tensor,target_tensor in mnist_train:
        C.train(image_data_tensor,target_tensor)
        pass
    pass
C.plot_progress()

#单个测试
mnist_test=ai.MnistDataset('python\pytorch\mnist_test_10.csv')
record=1
mnist_test.plot_image(record)
image_data=mnist_test[record][1]
output=C.forward(image_data)
pandas.DataFrame(output.detach().numpy()).plot(kind='bar',legend=False,ylim=(0,1))
matplotlib.pyplot.show()

#测试整个测试集
score=0
items=10

for label,image_data_tensor,target_tensor in mnist_test:
    answer=C.forward(image_data_tensor).detach().numpy()
    if(answer.argmax() == label):
        score += 1
        pass
print(f'准确率为{100*score/items}%')
