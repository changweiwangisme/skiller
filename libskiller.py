# -*- coding: utf-8 -*-
########################################################################
#  @file libskiller.py
#  @brief libskiller API, used for multidimensional time sequence learning and prediction  
#  @author Wei WANG@Waseda University, Tokyo, Japan
#  @e-mail changwei.wang@gmail.com
#  @date created at 2022/05/01
#
########################################################################


import torch
import torchvision
import numpy
import math
import matplotlib.pyplot as plt
import mpl_toolkits
#from mpl_toolkits.mplot3d import Axes3D
pi = math.pi




# global variable
DATASET_train = 0
DATALOADER_train = 0


# gpu check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# network
class Skiller_RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(Skiller_RNN, self).__init__()
        self.rnn = torch.nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.SEGMENT_N = 0
        self.POINT_K = 0
        self.data_feature = 0
    def forward(self, s):
        batch_size = s.size(0)
        #print("batch_size: ",batch_size)
        s = s.to(device)
        h0 = torch.zeros(self.n_layers, s.size(0), self.hidden_size).to(device)
        #s_rnn, hidden = self.rnn(s, h0)
        s_rnn, hidden = self.rnn(s, None)
        out = self.fc(s_rnn[:, -1, :])
        return out
    def set_sequence_parameters(self,segment_n,point_k):
        self.SEGMENT_N = segment_n
        self.POINT_K = point_k


# the function to load train data to the learning system
# path:         raw data file path
# segment_n:    sequence segment length 
# point_k:      sequence dimentional
# model:        the network model for learning
def load_traindata(path,segment_n,point_k,model):
    global DATASET_train
    global DATALOADER_train
    
    SEGMENT_N = segment_n
    POINT_K = point_k
        
    # read line
    #seq_list = []
    #point = numpy.zeros(POINT_K)
    #num = 0
    #line = f.readline()
    #while line:
    #    num+=1
    #    #print("num: ",num," line: ",line)
    #    for k in range(POINT_K):
    #        point[k] = line.split()[k]
    #    seq_list.append(point.copy())
    #    line = f.readline()
    #seq_matrix = numpy.array(seq_list)
    #print("read seq type: ", seq_matrix.dtype, " shape: ", seq_matrix.shape, " ndim: ", seq_matrix.ndim, " size: ", seq_matrix.size)
    
    # read lines
    f = open(path)
    lines = f.readlines()
    num = len(lines)
    #print("lines: ", len(lines))
    seq_matrix_raw = numpy.zeros((num,POINT_K))
    j = 0
    for line in lines:
        for k in range(POINT_K):
            seq_matrix_raw[j][k] = line.split()[k]
            #seq_matrix[j][k] = seq_matrix_raw[j][k]
        #print(seq_matrix[j,:])
        j+=1
    print("read seq raw type: ", seq_matrix_raw.dtype, " shape: ", seq_matrix_raw.shape, " ndim: ", seq_matrix_raw.ndim, " size: ", seq_matrix_raw.size)
    f.close()    
    
    # data feature
    data_feature = numpy.zeros((5,POINT_K))         # (num,min,max,mean,std)
    data_feature[0,:] = num
    data_feature[1,:] = numpy.min(seq_matrix_raw, axis=0)
    data_feature[2,:] = numpy.max(seq_matrix_raw, axis=0)
    data_feature[3,:] = numpy.mean(seq_matrix_raw, axis=0)
    data_feature[4,:] = numpy.std(seq_matrix_raw, axis=0)
    print("data feature[num,min,max,mean,std]: ")
    print(data_feature)
    
    # minmaxscalar
    seq_matrix = numpy.zeros((num,POINT_K))  
    for k in range(POINT_K):
        feature_min = data_feature[1][k]
        feature_max = data_feature[2][k]     
        feature_range = feature_max - feature_min
        if(0==feature_range):
            for i in range(num):
                seq_matrix[i][k] = 0
        else:
            for i in range(num):        
                seq_matrix[i][k] = (seq_matrix_raw[i][k] - feature_min)/feature_range
    
    # training data
    samples_NUM = num - SEGMENT_N
    INPUT_data = numpy.zeros((samples_NUM,SEGMENT_N,POINT_K))
    LABEL_data = numpy.zeros((samples_NUM,POINT_K))
    for i in range(samples_NUM):
        for j in range(SEGMENT_N):
            for k in range(POINT_K):
                INPUT_data[i][j][k] = seq_matrix[i+j][k]
        for k in range(POINT_K):
            LABEL_data[i][k] = seq_matrix[i+SEGMENT_N][k]
    INPUT_tensor = torch.FloatTensor(INPUT_data)
    LABEL_tensor = torch.FloatTensor(LABEL_data)

    # batch of training data
    DATASET_train = torch.utils.data.TensorDataset(INPUT_tensor,LABEL_tensor)
    DATALOADER_train = torch.utils.data.DataLoader(DATASET_train,batch_size=1,shuffle=True)
    
    # save to model
    model.SEGMENT_N = SEGMENT_N
    model.POINT_K = POINT_K
    model.data_feature = data_feature.copy()
    
    # show
    show_seq(seq_matrix_raw)
    

# the function to execute one training loop
# dataloader:   dataloader for training
# model:        the network model
# loss_fn:      loss function specified 
# optimizer:    optimizer specified
def train_loop(dataloader, model, loss_fn, optimizer):
    SEGMENT_N = model.SEGMENT_N
    POINT_K = model.POINT_K
    
    samples_num = len(dataloader.dataset)
    batches_num = len(dataloader)
    print("samples_num: ", samples_num, "batches_num: ", batches_num, "segment_N: ", SEGMENT_N, "point_K: ", POINT_K)
        
    model.train()
    loss_train = 0
    
    for batch, (s, out_label) in enumerate(dataloader):
        #print("batch: ",batch)
        
        # Compute prediction and loss
        out = model(s)
        loss = loss_fn(out, out_label.to(device))
        loss_train += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_train /= batch+1
    print("train: ", "loss: ",loss_train)


# the function to verify one training loop
# dataloader:   dataloader for training
# model:        the network model
# loss_fn:      loss function specified 
def verify_loop(dataloader, model, loss_fn):
    SEGMENT_N = model.SEGMENT_N
    POINT_K = model.POINT_K
    
    samples_num = len(dataloader.dataset)
    batches_num = len(dataloader)
    print("samples_num: ", samples_num, "batches_num: ", batches_num, "segment_N: ", SEGMENT_N, "point_K: ", POINT_K)

    model.eval()
    #with torch.no_grad():
    #    for batch, (s, y_label) in enumerate(dataloader):
    #        y = model(s)
       
    with torch.no_grad():
        seq_list = []
        point = numpy.zeros(POINT_K)
        
        s,out_label= dataloader.dataset[0]
        s = s.reshape(1,SEGMENT_N,POINT_K)
        
        for j in range(SEGMENT_N):
            for k in range(POINT_K):
                point[k] = s[0][j][k].item()
            seq_list.append(point.copy())
        
        for i in range(samples_num):
            s,out_label= dataloader.dataset[i]
            s = s.reshape(1,SEGMENT_N,POINT_K)
            out = model(s)
            for k in range(POINT_K):
                point[k] = out[0][k].item()
            seq_list.append(point.copy())

    print("verified seq length: ", len(seq_list))
    
    seq_matrix = numpy.array(seq_list)
    
    # show
    show_seq(seq_matrix)


# the function to predict the sequence from training data
# dataloader:   dataloader for training
# model:        the network model
# seq_len:      the sequence length that will be predicted 
def predict_loop(dataloader, model, seq_len):
    SEGMENT_N = model.SEGMENT_N
    POINT_K = model.POINT_K
    samples_num = len(dataloader.dataset)
    batches_num = len(dataloader)
    print("samples_num: ", samples_num, "batches_num: ", batches_num, "segment_N: ", SEGMENT_N, "point_K: ", POINT_K)
    
    seq_list = []
    point_predicted = numpy.zeros(POINT_K)
    
    model.eval()
    with torch.no_grad():
        s,out_label= dataloader.dataset[0]
        s = s.reshape(1,SEGMENT_N,POINT_K)
        out = model(s)

        for j in range(SEGMENT_N):
            for k in range(POINT_K):
                point_predicted[k] = s[0][j][k].item()
            seq_list.append(point_predicted.copy())
        
        for k in range(POINT_K):
            point_predicted[k] = out[0][k].item()
        seq_list.append(point_predicted.copy())
        
        for i in range(1,seq_len):
            #s_data = numpy.zeros((SEGMENT_N,POINT_K))
            s_data = numpy.array(seq_list[i:i+SEGMENT_N])
            s_tensor = torch.FloatTensor(s_data)
            s_tensor = s_tensor.reshape(1,SEGMENT_N,POINT_K)
            out_tensor = model(s_tensor)
            for k in range(POINT_K):
                point_predicted[k] = out_tensor[0][k].item()
            seq_list.append(point_predicted.copy())
        
    print("predicted seq length: ", len(seq_list))
    
    num = len(seq_list)
    seq_matrix = numpy.array(seq_list)
    
    # minmaxscalar back
    seq_matrix_raw = seq_matrix.copy() 
    data_feature = model.data_feature
    for k in range(POINT_K):
        feature_min = data_feature[1][k]
        feature_max = data_feature[2][k]     
        feature_range = feature_max - feature_min
        for i in range(num):        
            seq_matrix_raw[i][k] = feature_min + feature_range*seq_matrix[i][k] 
    
    # show
    show_seq(seq_matrix_raw)
    

# the function to predict a sequence from given start sequence, the start sequence are stored within a file 
# path:         start sequence file path
# model:        the network model
# seq_len:      the sequence length that will be predicted 
def predict_output(path, path_predicted, model, seq_len):
    SEGMENT_N = model.SEGMENT_N
    POINT_K = model.POINT_K
    
    # read lines
    f = open(path)
    lines = f.readlines()
    num = len(lines)
    #print("lines: ", len(lines))
    seq_matrix_raw = numpy.zeros((num,POINT_K))
    j = 0
    for line in lines:
        for k in range(POINT_K):
            seq_matrix_raw[j][k] = line.split()[k]
            #seq_matrix[j][k] = seq_matrix_raw[j][k]
        #print(seq_matrix[j,:])
        j+=1
    print("read seq raw type: ", seq_matrix_raw.dtype, " shape: ", seq_matrix_raw.shape, " ndim: ", seq_matrix_raw.ndim, " size: ", seq_matrix_raw.size)
    f.close()    
    
    # minmaxscalar
    data_feature = model.data_feature
    seq_matrix = numpy.zeros((num,POINT_K))  
    for k in range(POINT_K):
        feature_min = data_feature[1][k]
        feature_max = data_feature[2][k]     
        feature_range = feature_max - feature_min
        if(0==feature_range):
            for i in range(num):
                seq_matrix[i][k] = 0
        else:
            for i in range(num):        
                seq_matrix[i][k] = (seq_matrix_raw[i][k] - feature_min)/feature_range
    
    # start data
    samples_NUM = num - SEGMENT_N
    INPUT_data = numpy.zeros((samples_NUM,SEGMENT_N,POINT_K))
    LABEL_data = numpy.zeros((samples_NUM,POINT_K))
    for i in range(samples_NUM):
        for j in range(SEGMENT_N):
            for k in range(POINT_K):
                INPUT_data[i][j][k] = seq_matrix[i+j][k]
        for k in range(POINT_K):
            LABEL_data[i][k] = seq_matrix[i+SEGMENT_N][k]
    INPUT_tensor = torch.FloatTensor(INPUT_data)
    LABEL_tensor = torch.FloatTensor(LABEL_data)
    
    # predict
    seq_list = []
    point_predicted = numpy.zeros(POINT_K)
    
    model.eval()
    with torch.no_grad():
        s = INPUT_tensor[0]
        s = s.reshape(1,SEGMENT_N,POINT_K)
        out = model(s)

        for j in range(SEGMENT_N):
            for k in range(POINT_K):
                point_predicted[k] = s[0][j][k].item()
            seq_list.append(point_predicted.copy())
        
        for k in range(POINT_K):
            point_predicted[k] = out[0][k].item()
        seq_list.append(point_predicted.copy())
        
        for i in range(1,seq_len):
            #s_data = numpy.zeros((SEGMENT_N,POINT_K))
            s_data = numpy.array(seq_list[i:i+SEGMENT_N])
            s_tensor = torch.FloatTensor(s_data)
            s_tensor = s_tensor.reshape(1,SEGMENT_N,POINT_K)
            out_tensor = model(s_tensor)
            for k in range(POINT_K):
                point_predicted[k] = out_tensor[0][k].item()
            seq_list.append(point_predicted.copy())
        
    print("predicted seq length: ", len(seq_list))
    
    num = len(seq_list)
    seq_matrix = numpy.array(seq_list)
    
    # minmaxscalar back
    seq_matrix_raw = seq_matrix.copy() 
    data_feature = model.data_feature
    for k in range(POINT_K):
        feature_min = data_feature[1][k]
        feature_max = data_feature[2][k]     
        feature_range = feature_max - feature_min
        for i in range(num):        
            seq_matrix_raw[i][k] = feature_min + feature_range*seq_matrix[i][k] 
    
    # show
    show_seq(seq_matrix_raw)
    
    # save seq to file
    #strfilename = "mldata_predicted.txt"
    strfilename = path_predicted
    datafile = open(strfilename,"w")
    for i in range(len(seq_list)):
        strrow = ""
        for k in range(POINT_K):
            strrow = strrow + str(seq_matrix_raw[i][k]) + " "
        strrow = strrow  + "\r"
        datafile.write(strrow)
    datafile.close()


# the function to load multiple train data files as raw to the learning system (not used)
def load_traindata_multiple_raw(path,file_num,segment_n,point_k,model):
    global DATASET_train
    global DATALOADER_train
    
    SEGMENT_N = segment_n
    POINT_K = point_k
    
    seq_matrix_list = []
    INPUT_data_all = []
    LABEL_data_all = []
    
    for n in range(file_num):
        fullpath = ""
        fullpath = path + str(n+1) + ".txt" 
        f = open(fullpath)
        
        # read lines
        lines = f.readlines()
        num = len(lines)
        #print("lines: ", len(lines))
        seq_matrix = numpy.zeros((num,POINT_K))
        j = 0
        for line in lines:
            for k in range(POINT_K):
                seq_matrix[j][k] = line.split()[k]
            #print(seq_matrix[j,:])
            j+=1
        print("read seq type: ", seq_matrix.dtype, " shape: ", seq_matrix.shape, " ndim: ", seq_matrix.ndim, " size: ", seq_matrix.size)
        
        # training data
        samples_NUM = num - SEGMENT_N
        INPUT_data = numpy.zeros((samples_NUM,SEGMENT_N,POINT_K))
        LABEL_data = numpy.zeros((samples_NUM,POINT_K))
        for i in range(samples_NUM):
            for j in range(SEGMENT_N):
                for k in range(POINT_K):
                    INPUT_data[i][j][k] = seq_matrix[i+j][k]
            for k in range(POINT_K):
                LABEL_data[i][k] = seq_matrix[i+SEGMENT_N][k]
        
        show_seq(seq_matrix)
        seq_matrix_list.append(seq_matrix.copy())
        
        if(0==n):
            INPUT_data_all = INPUT_data.copy()
            LABEL_data_all = LABEL_data.copy()
        else:
            INPUT_data_all = numpy.append(INPUT_data_all,INPUT_data,axis = 0)
            LABEL_data_all = numpy.append(LABEL_data_all,LABEL_data,axis = 0)
        
        print("combined input data type: ", INPUT_data_all.dtype, " shape: ", INPUT_data_all.shape, " ndim: ", INPUT_data_all.ndim, " size: ", INPUT_data_all.size)

        
    INPUT_tensor = torch.FloatTensor(INPUT_data_all)
    LABEL_tensor = torch.FloatTensor(LABEL_data_all)
        
    # batch of training data
    DATASET_train = torch.utils.data.TensorDataset(INPUT_tensor,LABEL_tensor)
    DATALOADER_train = torch.utils.data.DataLoader(DATASET_train,batch_size=1,shuffle=True)    
    
    f.close()    


# the function to load multiple train data files to the learning system
# one file can represent the whole sequence of one experiment, the files names are orgainzed in a specified way.
# e.g, for path: "datafile", file_num: 3,  the following files will be loaded: "datafile1.txt", "datafile2.txt", "datafile3.txt"   
# path:         raw data file path
# file_num:     files number
# segment_n:    sequence segment length 
# point_k:      sequence dimentional
# model:        the network model for learning
def load_traindata_multiple(path,file_num,segment_n,point_k,model):
    global DATASET_train
    global DATALOADER_train
    
    SEGMENT_N = segment_n
    POINT_K = point_k
    
    seq_matrix_raw_all = 0
    seq_matrix_raw_list = []
    data_feature_list = []

    INPUT_data_all = 0
    LABEL_data_all = 0
    
    # read raw data from multiple file
    for n in range(file_num):
        fullpath = ""
        fullpath = path + str(n+1) + ".txt" 
        f = open(fullpath)
        
        # read lines
        lines = f.readlines()
        num = len(lines)
        #print("lines: ", len(lines))
        seq_matrix_raw = numpy.zeros((num,POINT_K))
        j = 0
        for line in lines:
            for k in range(POINT_K):
                seq_matrix_raw[j][k] = line.split()[k]
            #print(seq_matrix[j,:])
            j+=1
        print("read seq raw type: ", seq_matrix_raw.dtype, " shape: ", seq_matrix_raw.shape, " ndim: ", seq_matrix_raw.ndim, " size: ", seq_matrix_raw.size)
        
        f.close()    
        
        # data feature
        data_feature = numpy.zeros((5,POINT_K))         # (num,min,max,mean,std)
        data_feature[0,:] = num
        data_feature[1,:] = numpy.min(seq_matrix_raw, axis=0)
        data_feature[2,:] = numpy.max(seq_matrix_raw, axis=0)
        data_feature[3,:] = numpy.mean(seq_matrix_raw, axis=0)
        data_feature[4,:] = numpy.std(seq_matrix_raw, axis=0)
        print("data feature[num,min,max,mean,std]: ")
        print(data_feature)
    
        seq_matrix_raw_list.append(seq_matrix_raw.copy())
        data_feature_list.append(data_feature.copy())
        
        if(0==n):
            seq_matrix_raw_all = seq_matrix_raw.copy()
        else:
            seq_matrix_raw_all = numpy.append(seq_matrix_raw_all, seq_matrix_raw, axis = 0)
       
        show_seq(seq_matrix_raw)
        
    # data feature all
    data_feature_all = numpy.zeros((5,POINT_K))         # (num,min,max,mean,std)
    data_feature_all[0,:] = seq_matrix_raw_all.shape[0]
    data_feature_all[1,:] = numpy.min(seq_matrix_raw_all, axis=0)
    data_feature_all[2,:] = numpy.max(seq_matrix_raw_all, axis=0)
    data_feature_all[3,:] = numpy.mean(seq_matrix_raw_all, axis=0)
    data_feature_all[4,:] = numpy.std(seq_matrix_raw_all, axis=0)
    print("data feature all[num,min,max,mean,std]: ")
    print(data_feature_all)
    
    # create training data
    for n in range(file_num):
        seq_matrix_raw = seq_matrix_raw_list[n]
        #data_feature = data_feature_list[n]
        data_feature = data_feature_all
        num = seq_matrix_raw.shape[0]
        
        # minmaxscalar
        seq_matrix = numpy.zeros((num,POINT_K))  
        for k in range(POINT_K):
            feature_min = data_feature[1][k]
            feature_max = data_feature[2][k]     
            feature_range = feature_max - feature_min
            if(0==feature_range):
                for i in range(num):
                    seq_matrix[i][k] = 0
            else:
                for i in range(num):        
                    seq_matrix[i][k] = (seq_matrix_raw[i][k] - feature_min)/feature_range
        
        # training data
        samples_NUM = num - SEGMENT_N
        INPUT_data = numpy.zeros((samples_NUM,SEGMENT_N,POINT_K))
        LABEL_data = numpy.zeros((samples_NUM,POINT_K))
        for i in range(samples_NUM):
            for j in range(SEGMENT_N):
                for k in range(POINT_K):
                    INPUT_data[i][j][k] = seq_matrix[i+j][k]
            for k in range(POINT_K):
                LABEL_data[i][k] = seq_matrix[i+SEGMENT_N][k]
        
        #show_seq(seq_matrix)
        #seq_matrix_list.append(seq_matrix.copy())
        
        if(0==n):
            INPUT_data_all = INPUT_data.copy()
            LABEL_data_all = LABEL_data.copy()
        else:
            INPUT_data_all = numpy.append(INPUT_data_all,INPUT_data,axis = 0)
            LABEL_data_all = numpy.append(LABEL_data_all,LABEL_data,axis = 0)
        
        print("combined input data type: ", INPUT_data_all.dtype, " shape: ", INPUT_data_all.shape, " ndim: ", INPUT_data_all.ndim, " size: ", INPUT_data_all.size)
        
    INPUT_tensor = torch.FloatTensor(INPUT_data_all)
    LABEL_tensor = torch.FloatTensor(LABEL_data_all)
        
    # batch of training data
    DATASET_train = torch.utils.data.TensorDataset(INPUT_tensor,LABEL_tensor)
    DATALOADER_train = torch.utils.data.DataLoader(DATASET_train,batch_size=1,shuffle=True)    
    
    # save to model
    model.SEGMENT_N = SEGMENT_N
    model.POINT_K = POINT_K
    model.data_feature = data_feature_all.copy()


# the function to visualize a multidimensional sequence
# seq_matrix:      the matrix data structure for multidimensional sequence
def show_seq(seq_matrix):
    show_seq_2midpoints(seq_matrix)
    #show_seq_7joints_(seq_matrix)


# the function to visualize a multidimensional sequence of 2 midpoints
# seq_matrix:      the matrix data structure for multidimensional sequence
def show_seq_2midpoints(seq_matrix):
    x_P_seq = seq_matrix[:,0] 
    y_P_seq = seq_matrix[:,1] 
    z_P_seq = seq_matrix[:,2] 
    x_M_seq = seq_matrix[:,3] 
    y_M_seq = seq_matrix[:,4] 
    z_M_seq = seq_matrix[:,5] 

    fig = plt.figure()
    ax3d = mpl_toolkits.mplot3d.Axes3D(fig)
    
    ax3d.scatter(x_P_seq,y_P_seq,z_P_seq,c='g',marker='*')
    ax3d.scatter(x_M_seq,y_M_seq,z_M_seq,c='b',marker='*')

    #ax3d.set_xlim(-1, 1)
    #ax3d.set_ylim(-1, 1)
    #ax3d.set_zlim(-1, 1)
    ax3d.set_xlim(0, 600)
    ax3d.set_ylim(-300, 300)
    ax3d.set_zlim(-300, 300)
    #ax3d.set_xlim(-2000, 2000)
    #ax3d.set_ylim(-2000, 2000)
    #ax3d.set_zlim(-2000, 2000)
    ax3d.set_xlabel('X[mm]')
    ax3d.set_ylabel('Y[mm]')
    ax3d.set_zlabel('Z[mm]')
    plt.show()


# the function to visualize a multidimensional sequence of 7 joints
# seq_matrix:      the matrix data structure for multidimensional sequence
def show_seq_7joints(seq_matrix):
    j1_seq = seq_matrix[:,0] 
    j2_seq = seq_matrix[:,1] 
    j3_seq = seq_matrix[:,2] 
    j4_seq = seq_matrix[:,3] 
    j5_seq = seq_matrix[:,4] 
    j6_seq = seq_matrix[:,5] 
    j7_seq = seq_matrix[:,6] 
    
    #fig=plt.figure(num=1,figsize=(4,4))
    fig = plt.figure()
    ax=fig.add_subplot(111)
    
    #plt.plot(j1_seq,  label = "line 1", linestyle="-")
    ax.plot(j1_seq,  label = "joint 1", linestyle="--")
    ax.plot(j2_seq,  label = "joint 2", linestyle="--")
    ax.plot(j3_seq,  label = "joint 3", linestyle="--")
    ax.plot(j4_seq,  label = "joint 4", linestyle="--")
    ax.plot(j5_seq,  label = "joint 5", linestyle="--")
    ax.plot(j6_seq,  label = "joint 6", linestyle="--")
    ax.plot(j7_seq,  label = "joint 7", linestyle="--")
    
    ax.set_xlim(-10, 120)
    ax.set_ylim(-180, 180)
    ax.set_xlabel('time step')
    ax.set_ylabel('joint angle[degree]')
    
    ax.legend()
    plt.show()