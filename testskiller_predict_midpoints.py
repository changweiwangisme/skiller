# -*- coding: utf-8 -*-
########################################################################
#  @file testskiller_predict_midpoints.py
#  @brief this is a sample program for midpoints sequence prediction
#  @author Wei WANG@Waseda University, Tokyo, Japan
#  @e-mail changwei.wang@gmail.com
#  @date created at 2022/05/16
#
########################################################################


import torch
import torchvision
import libskiller as theskiller

print("skiller start")

# load model
mymodel = torch.load('mymodel.pth')

print("predict loop")
theskiller.predict_output("midpointdata_circle_noise5_start.txt","midpointdata_circle_noise5_predict.txt",mymodel,200)

print("skiller end")
