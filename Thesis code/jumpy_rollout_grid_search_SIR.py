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

#### load dataset 
with open("C:/Users/barts/università/Tesi magistrale/Datasets_Generation/Datasets_Generation/data/SIR/SIR_validation_set_64steps_ncp.pickle", 'rb') as file_handle:
    eSIR_data = pickle.load(file_handle)
#tdvae = tdvae.cuda()

dati_aux = eSIR_data['X'].reshape(50000,21,3)
somma=np.sum(dati_aux,axis=2)
#Y=np.repeat(eSIR_data_val['Y_s0'],repeats=2000,axis=0).reshape(50000,1,2)
#dati_aux = np.hstack([Y,dati_aux])
#dati=dati_aux/100
eSIR_data['X']=dati_aux/somma[...,None]

data=eSIR_Dataset(eSIR_data['X'])
batch_size = 6
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
idx, sequence = next(enumerate(data_loader))

diz_output={}
for belief_state_size in [10,50,100,200]:
    for state_size in [2,4,8,10,15]:
        #### load trained model
        checkpoint_2layer = torch.load("./output/model/new_model_epoch_99_belief_{}_state_{}_SIR.pt".format(belief_state_size,state_size))
        input_size = 3
        processed_x_size = 3
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
        t1, t2 = 5, 6
        t3=t1
        rollout_sequence = tdvae.jumpy_rollout(sequence, t1, t2)
        indici=sequence[1]
        s=somma[indici,0]
        diz_output['output_originale_belief_{}_state_{}'.format(belief_state_size,state_size)]=rollout_sequence.detach().numpy() * s[:, np.newaxis, np.newaxis]
        #print(np.round(prova))

        #deltat=3
        #roll=tdvae.rollout(sequence, t3, t3+deltat)[0]
        #prova_roll=roll.detach().numpy() * 100
        #t3=t3+deltat
        #for k in range(int(((t2-t1+1)/(deltat+1))-1)):
        #    roll=tdvae.rollout(sequence, t3, t3+deltat)[0]
        #    prova_roll=np.concatenate([prova_roll,roll.detach().numpy() * 100],axis=1)
        #    t3=t3+deltat
        #diz_output['output_spezzato_belief_{}_state_{}'.format(belief_state_size,state_size)]=prova_roll
        #print(np.round(prova_roll))


for i in range(batch_size):
    asse1=0
    fig, axs = plt.subplots(4,5,figsize=(25,20))
    #fig, axs = plt.subplots(2,3)
    for belief_state_size in [10,50,100,200]:
        asse2=0
        for state_size in [2,4,8,10,15]:
            predictions=diz_output['output_originale_belief_{}_state_{}'.format(belief_state_size,state_size)][i,:,:]
            original=dati_aux[indici[i],:,:]
           # pred_roll=diz_output['output_spezzato_belief_{}_state_{}'.format(belief_state_size,state_size)][i,:,:]
            
            S_or=original[:,0]
            I_or=original[:,1]
            R_or=original[:,2]
            S_pred=predictions[:,0]
            I_pred=predictions[:,1]
            R_pred=predictions[:,2]
            #S_roll=pred_roll[:,0]
    
    
            axs[asse1,asse2].plot(range(21),S_or,color='cyan',label='S')
            axs[asse1,asse2].axvline(t1, color='grey',linestyle='--')
            axs[asse1,asse2].plot(range(t2,22),S_pred,color='blue',label='S_pred')
            #axs[asse1,asse2].plot(range(t1-1,t2),S_roll,color='red',label='S_roll')
            axs[asse1,asse2].plot(range(21),I_or,color='orange',label='I')
            axs[asse1,asse2].plot(range(t2,22),I_pred,color='red',label='I_pred')
            axs[asse1,asse2].plot(range(21),R_or, color='pink',label='R')
            axs[asse1,asse2].plot(range(t2,22),R_pred,color='purple',label='R_pred')
            #axs[asse1,asse2].set_title('Belief = {}, state = {}'.format(belief_state_size,state_size))
            asse2=asse2+1
        asse1=asse1+1
            
    axs[0,4].legend(loc='upper right',bbox_to_anchor=(1.4, 1))
