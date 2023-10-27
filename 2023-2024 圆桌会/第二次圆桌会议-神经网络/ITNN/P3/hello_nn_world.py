# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 23:35:04 2023

@author: de-mi
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import time

class Net(torch.nn.Module):
#构造 
    def __init__(self): #what is "__init__"
        super().__init__()    # 
        self.fc1 = torch.nn.Linear(28*28,64)   #
        self.fc2 = torch.nn.Linear(64,64)
        self.fc3 = torch.nn.Linear(64,32)
        self.fc4 = torch.nn.Linear(32,10)
#前向
    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x),dim=1) #
        return x
#导入数据   
def get_data_loader(is_train):
        to_tensor = transforms.Compose([transforms.ToTensor()])#
        data_set  =MNIST("",is_train,transform=to_tensor,download=True)#
        return DataLoader(data_set,batch_size=64,shuffle=True)#
    
def evaluate(test_data,net):
        n_correct=0
        n_total  =0
        with torch.no_grad():
            for(x,y) in test_data:
                outputs = net.forward(x.view(-1,28*28))
                for i,output in enumerate(outputs):
                    if torch.argmax(output)==y[i]:
                        n_correct = n_correct + 1
                    n_total = n_total + 1
        return n_correct / n_total *100


train_data = get_data_loader(is_train=True)
test_data  = get_data_loader(is_train=False)
net=Net()
    
print("initial acc:", evaluate(test_data,net),'%')
time_start = time.time()

#训练网络    
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
for epoch in range(2):
        for (x,y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1,28*28))
            loss = torch.nn.functional.nll_loss(output,y)
            loss.backward()
            optimizer.step()
        print("epoch",epoch,"acc",evaluate(test_data,net),'%')
 
    
time_end = time.time()     
deltat= time_end - time_start   #运行所花时间
print('time cost', deltat, 's')
for (n,(x,_)) in enumerate(test_data):
        if n>5:
            break
        predict = torch.argmax(net.forward(x[0].view(-1,28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28,28))
        plt.title("prediction:"+str(int(predict)))
plt.show()
    


    
    





         
        

    