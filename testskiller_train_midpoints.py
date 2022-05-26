# -*- coding: utf-8 -*-
########################################################################
#  @file testskiller_train_midpoints.py
#  @brief this is a sample program for midpoints sequence training
#  @author Wei WANG@Waseda University, Tokyo, Japan
#  @e-mail changwei.wang@gmail.com
#  @date created at 2022/05/16
#
########################################################################

# 
import torch
import torchvision
import libskiller as theskiller


print("skiller train start")

# model
print("skiller: model")
mymodel = theskiller.Skiller_RNN(6,64*1,6,1*1).to(theskiller.device)
print(mymodel)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.01)

# load data
theskiller.load_traindata("mdata_circle_noise5.txt",10,6,mymodel)
#theskiller.load_traindata_multiple("datafile",4,10,6,mymodel)

# train
print("skiller: train")

for i in range(201):
    print("epoch: ", i)
    print("train loop")
    theskiller.train_loop(theskiller.DATALOADER_train, mymodel, criterion, optimizer)
    if (i%20 == 0):
        print("predict loop")
        theskiller.predict_loop(theskiller.DATALOADER_train, mymodel,75)

torch.save(mymodel, 'mymodel.pth')

print("skiller train end")
