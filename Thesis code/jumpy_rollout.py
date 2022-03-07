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
checkpoint_2layer = torch.load("./output/model/new_model_epoch_99_belief_50_state_2_eSIR.pt")
input_size = 2
processed_x_size = 2
belief_state_size = 50
state_size = 2
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint_2layer['model_state_dict'])
optimizer.load_state_dict(checkpoint_2layer['optimizer_state_dict'])

#### load dataset 
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/eSIRS/eSIRS_validation_set_fixed_param.pickle", 'rb') as file_handle:
    eSIR_data = pickle.load(file_handle)
tdvae.eval()
#tdvae = tdvae.cuda()

somma=np.sum(eSIR_data['X'],axis=2)
dati_aux = eSIR_data['X'].reshape(50000,32,2)
dati=dati_aux/100

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
    #s=somma[indici,0]
    prova[k]=rollout_sequence.detach().numpy() * 100
    #print(np.round(prova))
print("Time to generate the training set w fixed param =", time.time()-start_time)

#deltat=3
#roll=tdvae.rollout(sequence, t3, t3+deltat)[0]
#prova_roll=roll.detach().numpy() * 100
#t3=t3+deltat
#for k in range(int(((t2-t1+1)/(deltat+1))-1)):
#    roll=tdvae.rollout(sequence, t3, t3+deltat)[0]
#    prova_roll=np.concatenate([prova_roll,roll.detach().numpy() * 100],axis=1)
#    t3=t3+deltat
#print(np.round(prova_roll))

dist_base = [[None for aa in range(12) ] for bb in range(batch_size)]
for i in range(batch_size):
    predictions = [[None] * 12 ] * batch_size
    original=dati_aux[indici[i],1:,:]
    for q in range(12):
        predictions[q]=prova[q][i,:,:]
        dist_base[i][q] = ((original - predictions[q])**2)#.mean(axis=0)

distanze = [None for cc in range(batch_size)]
for j in range(batch_size):
    distanze[j] = np.dstack(dist_base[j]).mean(axis=2)
    
distanza = np.dstack(distanze).mean(axis=2)

plt.plot(distanza[:,0], color='blue',label='S')    
plt.plot(distanza[:,1], color='red',label='I') 
plt.legend(loc='upper right',bbox_to_anchor=(1.4, 1))
plt.show()


'''
fig, axs = plt.subplots(2,3, figsize=(15,10))
for i in range(batch_size):
    predictions = [None] * 10
    for q in range(10):
        predictions[q]=prova[q][i,:,:]
    original=dati_aux[indici[i],:,:]
    #pred_roll=prova_roll[i,:,:]
    
    S_or=original[:,0]
    I_or=original[:,1]
    #R_or=original[:,2]
    S_pred = [None] * 10
    I_pred = [None] * 10
    for j in range(10):
        S_pred[j]=predictions[j][:,0]
        I_pred[j]=predictions[j][:,1]
    #R_pred=predictions[:,2]
    #S_roll=pred_roll[:,0]
    
    
    axs[i%2,i%3].plot(range(32),S_or,color='cyan',label='S', linewidth=2)
    axs[i%2,i%3].axvline(t1, color='grey',linestyle='--')
    axs[i%2,i%3].plot(range(t2,33),S_pred[9],color='blue',label='S_pred', linewidth =.5)
    for h in range(9):
        axs[i%2,i%3].plot(range(t2,33),S_pred[h],color='blue', linewidth =.5)
        axs[i%2,i%3].plot(range(t2,33),I_pred[h],color='red', linewidth=.5)
    #axs[i%2,i%3].plot(range(t1-1,t2),S_roll,color='red',label='S_roll')
    axs[i%2,i%3].plot(range(32),I_or,color='orange',label='I', linewidth=2)
    axs[i%2,i%3].plot(range(t2,33),I_pred[9],color='red',label='I_pred', linewidth=.5)
    #axs[i%2,i%3].plot(range(64),R_or, color='pink',label='R')
    #axs[i%2,i%3].plot(range(t1-1,t2),R_pred,color='purple',label='R_pred')

axs[0,2].legend(loc='upper right',bbox_to_anchor=(1.8, 0.5))
    #plt.show()
'''