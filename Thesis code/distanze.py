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


'''
#### load trained model
checkpoint_2layer = torch.load("./output/model/new_model_epoch_449_lstm2layer_decoder_dblock_SI_random.pt")
input_size = 2
processed_x_size = 2
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint_2layer['model_state_dict'])
optimizer.load_state_dict(checkpoint_2layer['optimizer_state_dict'])
'''
#### load dataset 
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/eSIRS/eSIRS_validation_set_fixed_param.pickle", 'rb') as file_handle:
    eSIR_data = pickle.load(file_handle)
#tdvae.eval()
#tdvae = tdvae.cuda()

dati_aux = eSIR_data['X'].reshape(50000,32,2)
somma=np.sum(dati_aux,axis=2)
dati=dati_aux/100
#dati=dati_aux/somma[...,None]

data = eSIR_Dataset(dati)#[:,:,[0]])
batch_size = 50
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
idx, sequence = next(enumerate(data_loader))

diz_sim_esir={}
for belief_state_size in [10,50,100,200]:
    for state_size in [2,4,8,10,15]:
        #### load trained model
        checkpoint_2layer = torch.load("C:/Users/barts/università/Tesi magistrale/debug/output/model/new_model_epoch_99_belief_{}_state_{}_eSIR.pt".format(belief_state_size,state_size))#,map_location=torch.device('cpu'))
        input_size = 2
        processed_x_size = 2
        #belief_state_size = 50
        #state_size = 8
        tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
        optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)
        
        tdvae.load_state_dict(checkpoint_2layer['model_state_dict'])
        optimizer.load_state_dict(checkpoint_2layer['optimizer_state_dict'])
        tdvae.eval()



        #images = images.cuda()

        ## calculate belief
        tdvae.forward(sequence)

        ## jumpy rollout
        t1, t2 = 1,2
        rollout_sequence = tdvae.jumpy_rollout(sequence, t1, t2)
        indici=sequence[1]
        #s=somma[indici,0]
        prova = [None]*50
        for k in range(50):
            rollout_sequence = tdvae.jumpy_rollout(sequence, t1, t2)
            indici=sequence[1]
            s=somma[indici,0]
            prova[k]=rollout_sequence.detach().numpy() * 100# s[:, np.newaxis, np.newaxis]
        #diz_sim_esir['esir_belief_{}_state_{}'.format(belief_state_size,state_size)]=rollout_sequence.detach().numpy() * 100
        #print(np.round(prova))
        
        dist_base = [[None for aa in range(50) ] for bb in range(batch_size)]
        for i in range(batch_size):
            predictions = [[None] * 50 ] * batch_size
            original=dati_aux[indici[i],1:,:]
            for q in range(50):
                predictions[q]=prova[q][i,:,:]
                dist_base[i][q] = ((original - predictions[q])**2).mean(axis=0)

        distanze = [None for cc in range(batch_size)]
        for j in range(batch_size):
            distanze[j] = np.vstack(dist_base[j]).mean(axis=0)
    
        diz_sim_esir['esir_belief_{}_state_{}'.format(belief_state_size,state_size)] = np.vstack(distanze).mean(axis=0)
   


with open('C:/Users/barts/università/Tesi magistrale/simulazioni distanze/diz_sim_esir1.pkl', 'wb') as outfile:
   pickle.dump( diz_sim_esir, outfile)
#images = images.cuda()
'''
## calculate belief
#start_time = time.time()
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
#print("Time to generate the training set w fixed param =", time.time()-start_time)

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
        dist_base[i][q] = ((original - predictions[q])**2).mean(axis=0)

distanze = [None for cc in range(batch_size)]
for j in range(batch_size):
    distanze[j] = np.vstack(dist_base[j]).mean(axis=0)
    
distanza = np.vstack(distanze).mean(axis=0)
'''