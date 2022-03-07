import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_eSIR import *
from prep_data_eSIR import *
import sys

#### preparing dataset
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/Clock/Clock_training_set_dt=1_32steps.pickle", 'rb') as file_handle:
    eSIR_data = pickle.load(file_handle)
    
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/Clock/Clock_validation_set_dt=1_32steps.pickle", 'rb') as file_handle:
    eSIR_data_val = pickle.load(file_handle)

eSIR_data['Y_s0']=eSIR_data['Y_s0'].reshape(100000,1,3)
eSIR_data_1 = np.hstack([eSIR_data['Y_s0'],eSIR_data['X']])
somma=np.sum(eSIR_data_1,axis=2)
eSIR_data_1=eSIR_data_1/somma[...,None]

data = eSIR_Dataset(eSIR_data_1[:,:,[0]])#l'ultima quadra è una prova

dati_aux = eSIR_data_val['X'].reshape(50000,32,3)
#somma_val=np.sum(dati_aux,axis=2)
Y=np.repeat(eSIR_data_val['Y_s0'],repeats=2000,axis=0).reshape(50000,1,3)
dati_aux = np.hstack([Y,dati_aux])
somma_val=np.sum(dati_aux,axis=2)
#dati=dati_aux/100
eSIR_data_val['X']=dati_aux/somma_val[...,None]

val_data=eSIR_Dataset(eSIR_data_val['X'][:,:,[0]])

batch_size = 512
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)

val_data_loader = DataLoader(val_data,
                         batch_size = batch_size,
                         shuffle = True)

#### build a TD-VAE model
input_size = 1
processed_x_size = 1
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
#tdvae = tdvae.cuda()

#### training
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)
num_epoch = 4000
log_file_handle = open("./log/loginfo_new_decoder_dblock_clock_A_random.txt", 'w')
for epoch in range(num_epoch):
    train_loss = 0
    val_loss = 0
    cont=0
    cont_val=0
    for idx, sequence in enumerate(data_loader):        
        #sequence = sequence.cuda()       
        tdvae.forward(sequence)
        t_1 = np.random.choice(28)
        t_2 = t_1 + np.random.choice([1,2,3,4,5])
        loss = tdvae.calculate_loss(t_1, t_2)[0]
        dizionario_100_epoch_2_layer_decoder_dblock_clock_A_random=tdvae.calculate_loss(t_1, t_2)[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
        cont = cont + 1

        #print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
        #      file = log_file_handle, flush = True)
        
        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()))
        
    tdvae.eval()
    with torch.no_grad():
        for idx, sequence in enumerate(val_data_loader):
            t_1 = np.random.choice(28)
            t_2 = t_1 + np.random.choice([1,2,3,4,5])
            val_loss = tdvae.calculate_loss(t_1, t_2)[0]
            optimizer.requires_grad=True
            val_loss = val_loss + val_loss.item()
            cont_val = cont_val + 1
            
    val_loss=val_loss/cont_val
    train_loss=train_loss/cont
    print("epoch: {:>4d}, train_loss: {:.2f}, val_loss: {:.2f}".format(epoch, train_loss, val_loss))
    print("epoch: {:>4d}, train_loss: {:.2f}, val_loss: {:.2f}".format(epoch, train_loss, val_loss),
              file = log_file_handle, flush = True)

    if (epoch + 1) % 50 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': tdvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, "./output/model/new_model_epoch_{}_lstm2layer_decoder_dblock_clock_A_random.pt".format(epoch))
