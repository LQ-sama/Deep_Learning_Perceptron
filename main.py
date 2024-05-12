import torch 
from matplotlib import pyplot as plt
from torch import nn
from d2l import torch as d2l
import dataset as ppd
import math
import numpy
d2l.use_svg_display()
device = torch.device('cuda')
train_dataset = ppd.mnist_train
test_dataset = ppd.mnist_test
train_dataloader = ppd.train_dataloader
test_dataloader = ppd.test_dataloader
num_inputs = 784
num_outputs = 10
num_hiddens1 = 128
num_hiddens2 = 56

the_img_we_look = test_dataset[4][0].squeeze()
plt.imshow(the_img_we_look, cmap="gray")
plt.axis("off")
plt.show()

simple_net = nn.Sequential(nn.Flatten(),nn.Linear(num_inputs,num_hiddens1),nn.ReLU(),nn.Dropout(0.0),
                           nn.Linear(num_hiddens1,num_hiddens2),nn.ReLU(),nn.Dropout(0.0),
                           nn.Linear(num_hiddens2,num_outputs))

w1 = nn.Parameter(torch.randn(num_inputs,num_hiddens1,requires_grad=True)*0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens1,requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hiddens1,num_hiddens2,requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_hiddens2,requires_grad=True))
w3 = nn.Parameter(torch.randn(num_hiddens2,num_outputs,requires_grad=True)*0.01)
b3 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
print(w1)
params = [w1,b1,w2,b2 ,w3 ,b3]
##实现ReLU激活函数
def relu(x):
    re = torch.zeros_like(x)
    return torch.max(re, x)

def net(x):
    x= x.reshape(-1,num_inputs)
    hidden1 = relu(x@w1 + b1)
    hidden2 = relu(hidden1@w2 + b2)
    out = hidden2@w3 + b3
    return out
def what_is_going_on(x) :
    b11 = b1.data
    b22 = b2.data
    w11 = w1.data
    w22 = w2.data
    x = x.squeeze()
    x = x.reshape(-1,num_inputs)
    hidden1 = relu(x@w11+b11)
    the_hidden1_out = hidden1.reshape(16, 8)
    plt.imshow(the_hidden1_out, cmap="gray")
    plt.axis("off")
    plt.show()
    print(hidden1.shape)
    hidden2 = relu(hidden1@w22 + b22)
    the_hidden2_out = hidden2.reshape(7, 8)
    plt.imshow(the_hidden2_out, cmap="gray")
    plt.axis("off")
    plt.show()
    print(hidden2.shape)


loss = nn.CrossEntropyLoss(reduction='none')

num_epochs = 10
lr = 0.1
updater = torch.optim.SGD(params,lr=lr)
simple_net.train()
d2l.train_ch3(net,train_dataloader,test_dataloader,loss,num_epochs,updater)
simple_net.eval()
d2l.predict_ch3(net,test_dataloader)
d2l.plt.show()
what_is_going_on(test_dataset[4][0])

