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
checkpoint_2layer = torch.load("./output/model/new_model_epoch_99_belief_400_state_4_clock.pt",map_location=torch.device('cpu'))
input_size = 3
processed_x_size = 3
belief_state_size = 400
state_size = 4
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint_2layer['model_state_dict'])
optimizer.load_state_dict(checkpoint_2layer['optimizer_state_dict'])

#### load dataset 
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/Clock/Clock_validation_set_dt=1_32steps.pickle", 'rb') as file_handle:
    eSIR_data = pickle.load(file_handle)
tdvae.eval()
#tdvae = tdvae.cuda()

dati_aux = eSIR_data['X'].reshape(50000,32,3)
somma=np.sum(dati_aux,axis=2)
#dati=dati_aux/100
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
prova = [None]*50
for k in range(50):
    rollout_sequence = tdvae.jumpy_rollout(sequence, t1, t2)
    indici=sequence[1]
    s=somma[indici,0]
    prova[k]=rollout_sequence.detach().numpy() * s[:, np.newaxis, np.newaxis]
    #print(np.round(prova))
print("Time to generate the training set w fixed param =", time.time()-start_time)


dist_base = [[None for aa in range(50) ] for bb in range(batch_size)]
for i in range(batch_size):
    predictions = [[None] * 50 ] * batch_size
    original=dati_aux[indici[i],1:,:]
    for q in range(50):
        predictions[q]=prova[q][i,:,:]
        dist_base[i][q] = ((original - predictions[q])**2)#.mean(axis=0)

distanze = [None for cc in range(batch_size)]
for j in range(batch_size):
    distanze[j] = np.dstack(dist_base[j]).mean(axis=2)
    
distanza = np.dstack(distanze).mean(axis=2)

plt.plot(distanza[:,0], color='blue',label='A')    
plt.plot(distanza[:,1], color='red',label='B')    
plt.plot(distanza[:,2], color='purple',label='C') 
plt.legend(loc='upper right',bbox_to_anchor=(1.4, 1))
plt.show()