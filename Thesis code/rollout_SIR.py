import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model_eSIR import *
from prep_data_eSIR import *
import time



#### load trained model
checkpoint_2layer = torch.load("./output/model/new_model_epoch_399_lstm2layer_decoder_dblock_SIR_random.pt")
input_size = 3
processed_x_size = 3
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint_2layer['model_state_dict'])
optimizer.load_state_dict(checkpoint_2layer['optimizer_state_dict'])

#### load dataset 
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/SIR/SIR_validation_set_64steps_ncp.pickle", 'rb') as file_handle:
    eSIR_data = pickle.load(file_handle)
tdvae.eval()
#tdvae = tdvae.cuda()

dati_aux = eSIR_data['X'].reshape(50000,21,3)
somma=np.sum(dati_aux,axis=2)
dati=dati_aux/somma[...,None]

data = eSIR_Dataset(dati)#[:,:,[0]])
batch_size = 50
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
idx, sequence = next(enumerate(data_loader))

#images = images.cuda()

## calculate belief
start_time = time.time()
tdvae.forward(sequence)

## jumpy rollout
t1, t2 = 1, 2
#t3=t1
prova = [None]*12
for k in range(12):
    rollout_sequence = tdvae.jumpy_rollout(sequence, t1, t2)
    indici=sequence[1]
    s=somma[indici,0]
    prova[k]=rollout_sequence.detach().numpy() *  s[:, np.newaxis, np.newaxis]
    #print(np.round(prova))

print("Time to generate the training set w fixed param =", time.time()-start_time)
#rollout_sequence = tdvae.jumpy_rollout(sequence, t1, t2)
#indici=sequence[1]
#s=somma[indici,0]
#prova=rollout_sequence.detach().numpy() * s[:, np.newaxis, np.newaxis]
#print(np.round(prova))

#deltat=3
#roll=tdvae.rollout(sequence, t3, t3+deltat)[0]
#prova_roll=roll.detach().numpy() * 100
#t3=t3+deltat
#for k in range(int(((t2-t1+1)/(deltat+1))-1)):
#    roll=tdvae.rollout(sequence, t3, t3+deltat)[0]
#    prova_roll=np.concatenate([prova_roll,roll.detach().numpy() * 100],axis=1)
#    t3=t3+deltat
#print(np.round(prova_roll))

fig, axs = plt.subplots(2,3, figsize=(15,10))
for i in range(batch_size):
    predictions = [None] * 10
    for q in range(10):
        predictions[q]=prova[q][i,:,:]
    #predictions=prova[i,:,:]
    original=dati_aux[indici[i],:,:]
    #pred_roll=prova_roll[i,:,:]
    
    S_or=original[:,0]
    I_or=original[:,1]
    R_or=original[:,2]
    S_pred = [None] * 10
    I_pred = [None] * 10
    R_pred = [None] * 10
    for j in range(10):
        S_pred[j]=predictions[j][:,0]
        I_pred[j]=predictions[j][:,1]
        R_pred[j]=predictions[j][:,2]
    #S_pred=predictions[:,0]
    #I_pred=predictions[:,1]
    #R_pred=predictions[:,2]
    #S_roll=pred_roll[:,0]
    
    
    axs[i%2,i%3].plot(range(21),S_or,color='cyan',label='S',linewidth=2)
    axs[i%2,i%3].axvline(t1, color='grey',linestyle='--')
    axs[i%2,i%3].plot(range(t2,22),S_pred[9],color='blue',label='S_pred',linewidth=.5)
    #axs[i%2,i%3].plot(range(t1-1,t2),S_roll,color='red',label='S_roll')
    axs[i%2,i%3].plot(range(21),I_or,color='orange',label='I',linewidth=2)
    axs[i%2,i%3].plot(range(t2,22),I_pred[9],color='red',label='I_pred',linewidth=.5)
    axs[i%2,i%3].plot(range(21),R_or, color='pink',label='R',linewidth=2)
    axs[i%2,i%3].plot(range(t2,22),R_pred[9],color='purple',label='R_pred',linewidth=.5)
    for h in range(9):
        axs[i%2,i%3].plot(range(t2,22),S_pred[h],color='blue', linewidth =.5)
        axs[i%2,i%3].plot(range(t2,22),I_pred[h],color='red', linewidth=.5)
        axs[i%2,i%3].plot(range(t2,22),R_pred[h],color='purple',linewidth=.5)

axs[0,2].legend(loc='upper right',bbox_to_anchor=(1.8, 0.5))
    #plt.show()

