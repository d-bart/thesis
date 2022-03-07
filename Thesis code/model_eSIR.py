import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return mu, logsigma

class PreProcess(nn.Module):
    def __init__(self, input_size, processed_x_size):
        super(PreProcess, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t

class Decoder(nn.Module):
    def __init__(self, z_size, hidden_size, x_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)
        
    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = torch.sigmoid(self.fc3(t))
        return p

    
class TD_VAE(nn.Module):
    def __init__(self, x_size, processed_x_size, b_size, z_size):
        super(TD_VAE, self).__init__()
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.b_size = b_size
        self.z_size = z_size

        ## input pre-process layer
        #self.process_x = PreProcess(self.x_size, self.processed_x_size)
        
        ## one layer LSTM for aggregating belief states
        ## One layer LSTM is used here and I am not sure how many layers
        ## are used in the original paper from the paper.
        self.lstm = nn.LSTM(input_size = self.x_size,
                            hidden_size = self.b_size,
                            num_layers=2,
                            batch_first = True)

        ## Two layer state model is used. Sampling is done by sampling
        ## higher layer first.
        ## belief to state (b to z)
        ## (this is corresponding to P_B distribution in the reference;
        ## weights are shared across time but not across layers.)
        self.l2_b_to_z = DBlock(b_size, 50, z_size) # layer 2
        self.l1_b_to_z = DBlock(b_size + z_size, 50, z_size) # layer 1

        ## Given belief and state at time t2, infer the state at time t1
        ## infer state
        self.l2_infer_z = DBlock(b_size + 2*z_size + 1, 50, z_size) # layer 2
        self.l1_infer_z = DBlock(b_size + 2*z_size + z_size + 1, 50, z_size) # layer 1

        ## Given the state at time t1, model state at time t2 through state transition
        ## state transition
        self.l2_transition_z = DBlock(2*z_size + 1, 50, z_size)
        self.l1_transition_z = DBlock(2*z_size + z_size + 1, 50, z_size)

        ## state to observation
        self.z_to_x = DBlock(2*z_size,50,x_size) #Decoder(2*z_size, 200, x_size)


    def forward(self, sequence):        
        self.batch_size = sequence[0].size()[0]
        self.x = sequence[0]
        ## pre-precess image x
        #self.processed_x = self.process_x(self.x)

        ## aggregate the belief b
        self.b, (h_n, c_n) = self.lstm(self.x)
        
    def calculate_loss(self, t1, t2):
        batch_size = self.batch_size
        deltat = np.array([t2 - t1]*batch_size).reshape(batch_size,1)
        deltat = torch.tensor(deltat)

        ## Because the loss is based on variational inference, we need to
        ## draw samples from the variational distribution in order to estimate
        ## the loss function.
        
        ## sample a state at time t2 (see the reparametralization trick is used)
        ## z in layer 2
        t2_l2_z_mu, t2_l2_z_logsigma = self.l2_b_to_z(self.b[:, t2, :])
        t2_l2_z_epsilon = torch.randn_like(t2_l2_z_mu)
        t2_l2_z = t2_l2_z_mu + torch.exp(t2_l2_z_logsigma)*t2_l2_z_epsilon
        
        pb_t2_l2 = torch.distributions.Normal(t2_l2_z_mu, torch.exp(t2_l2_z_logsigma))

        ## z in layer 1
        t2_l1_z_mu, t2_l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t2,:], t2_l2_z),dim = -1))
        t2_l1_z_epsilon = torch.randn_like(t2_l1_z_mu)
        t2_l1_z = t2_l1_z_mu + torch.exp(t2_l1_z_logsigma)*t2_l1_z_epsilon
        
        pb_t2_l1 = torch.distributions.Normal(t2_l1_z_mu, torch.exp(t2_l1_z_logsigma))

        ## concatenate z from layer 1 and layer 2 
        t2_z = torch.cat((t2_l1_z, t2_l2_z), dim = -1)

        ## sample a state at time t1
        ## infer state at time t1 based on states at time t2
        t1_l2_qs_z_mu, t1_l2_qs_z_logsigma = self.l2_infer_z(
            torch.cat((self.b[:,t1,:], t2_z, deltat), dim = -1))
        t1_l2_qs_z_epsilon = torch.randn_like(t1_l2_qs_z_mu)
        t1_l2_qs_z = t1_l2_qs_z_mu + torch.exp(t1_l2_qs_z_logsigma)*t1_l2_qs_z_epsilon

        t1_l1_qs_z_mu, t1_l1_qs_z_logsigma = self.l1_infer_z(
            torch.cat((self.b[:,t1,:], t2_z, t1_l2_qs_z, deltat), dim = -1))
        t1_l1_qs_z_epsilon = torch.randn_like(t1_l1_qs_z_mu)
        t1_l1_qs_z = t1_l1_qs_z_mu + torch.exp(t1_l1_qs_z_logsigma)*t1_l1_qs_z_epsilon

        t1_qs_z = torch.cat((t1_l1_qs_z, t1_l2_qs_z), dim = -1)
        
        qs_t1_l2=torch.distributions.Normal(t1_l2_qs_z_mu, torch.exp(t1_l2_qs_z_logsigma))
        qs_t1_l1=torch.distributions.Normal(t1_l1_qs_z_mu, torch.exp(t1_l1_qs_z_logsigma))


        #### After sampling states z from the variational distribution, we can calculate
        #### the loss.

        ## state distribution at time t1 based on belief at time 1
        t1_l2_pb_z_mu, t1_l2_pb_z_logsigma = self.l2_b_to_z(self.b[:, t1, :])        
        t1_l1_pb_z_mu, t1_l1_pb_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t1,:], t1_l2_qs_z),dim = -1))
        
        pb_t1_l2=torch.distributions.Normal(t1_l2_pb_z_mu, torch.exp(t1_l2_pb_z_logsigma))
        pb_t1_l1=torch.distributions.Normal(t1_l1_pb_z_mu, torch.exp(t1_l1_pb_z_logsigma))

        ## state distribution at time t2 based on states at time t1 and state transition
        t2_l2_t_z_mu, t2_l2_t_z_logsigma = self.l2_transition_z(torch.cat((t1_qs_z, deltat),dim = -1))
        t2_l1_t_z_mu, t2_l1_t_z_logsigma = self.l1_transition_z(
            torch.cat((t1_qs_z, t2_l2_z, deltat), dim = -1))
        
        pt_t2_l2 = torch.distributions.Normal(t2_l2_t_z_mu,torch.exp(t2_l2_t_z_logsigma))
        pt_t2_l1 = torch.distributions.Normal(t2_l1_t_z_mu,torch.exp(t2_l1_t_z_logsigma))
        
        ## observation distribution at time t2 based on state at time t2
        t2_x_mu, t2_x_logsigma = self.z_to_x(t2_z)
        pd = torch.distributions.Normal(t2_x_mu,torch.exp(t2_x_logsigma))
        #t2_x_prob = self.z_to_x(t2_z)

        #### start calculating the loss

        #### KL divergence between z distribution at time t1 based on variational distribution
        #### (inference model) and z distribution at time t1 based on belief.
        #### This divergence is between two normal distributions and it can be calculated analytically
        
        ## KL divergence between t1_l2_pb_z, and t1_l2_qs_z
        loss = 0.5*torch.sum(((t1_l2_pb_z_mu - t1_l2_qs_z)/torch.exp(t1_l2_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l2_pb_z_logsigma, -1) - torch.sum(t1_l2_qs_z_logsigma, -1)
               
        L_t1_l2 = torch.distributions.kl_divergence(qs_t1_l2, pb_t1_l2)

        ## KL divergence between t1_l1_pb_z and t1_l1_qs_z
        loss += 0.5*torch.sum(((t1_l1_pb_z_mu - t1_l1_qs_z)/torch.exp(t1_l1_pb_z_logsigma))**2,-1) + \
               torch.sum(t1_l1_pb_z_logsigma, -1) - torch.sum(t1_l1_qs_z_logsigma, -1)  
               
        L_t1_l1 = torch.distributions.kl_divergence(qs_t1_l1, pb_t1_l1)
        
        #### The following four terms estimate the KL divergence between the z distribution at time t2
        #### based on variational distribution (inference model) and z distribution at time t2 based on transition.
        #### In contrast with the above KL divergence for z distribution at time t1, this KL divergence
        #### can not be calculated analytically because the transition distribution depends on z_t1, which is sampled
        #### after z_t2. Therefore, the KL divergence is estimated using samples
        
        ## state log probabilty at time t2 based on belief
        loss += torch.sum(-0.5*t2_l2_z_epsilon**2 - 0.5*t2_l2_z_epsilon.new_tensor(2*np.pi) - t2_l2_z_logsigma, dim = -1) 
        loss += torch.sum(-0.5*t2_l1_z_epsilon**2 - 0.5*t2_l1_z_epsilon.new_tensor(2*np.pi) - t2_l1_z_logsigma, dim = -1)

        ## state log probabilty at time t2 based on transition
        loss += torch.sum(0.5*((t2_l2_z - t2_l2_t_z_mu)/torch.exp(t2_l2_t_z_logsigma))**2 + 0.5*t2_l2_z.new_tensor(2*np.pi) + t2_l2_t_z_logsigma, -1)
        loss += torch.sum(0.5*((t2_l1_z - t2_l1_t_z_mu)/torch.exp(t2_l1_t_z_logsigma))**2 + 0.5*t2_l1_z.new_tensor(2*np.pi) + t2_l1_t_z_logsigma, -1)
        
        L_t2_l2 = pb_t2_l2.log_prob(t2_l2_z) - pt_t2_l2.log_prob(t2_l2_z)
        
        L_t2_l1 = pb_t2_l1.log_prob(t2_l1_z) - pt_t2_l1.log_prob(t2_l1_z)

        
        ## observation prob at time t2
        #loss += -torch.sum(self.x[:,t2,:]*torch.log(t2_x_prob) + (1-self.x[:,t2,:])*torch.log(1-t2_x_prob), -1)
        loss = torch.mean(loss)
        
        L_x = -pd.log_prob(self.x[:,t2,:])
        
        Loss = torch.sum(L_t1_l2,-1) + torch.sum(L_t1_l1,-1) + torch.sum(L_t2_l2,-1) + torch.sum(L_t2_l1,-1) + torch.sum(L_x,-1)
        Loss = torch.mean(Loss)
        
        dict_var={
            'belief_state':self.b,
            't2_l2_z_mu':t2_l2_z_mu,
            't2_l2_z_logsigma':torch.exp(t2_l2_z_logsigma),
            't2_l2_z_epsilon':t2_l2_z_epsilon,
            't2_l2_z':t2_l2_z,
            't2_l1_z_mu':t2_l1_z_mu,
            't2_l1_z_logsigma':torch.exp(t2_l1_z_logsigma),
            't2_l1_z':t2_l1_z,
            't2_z':t2_z,
            't1_l2_qs_z_mu':t1_l2_qs_z_mu,
            't1_l2_qs_z_logsigma':torch.exp(t1_l2_qs_z_logsigma),
            't1_l2_qs_z_epsilon':t1_l2_qs_z_epsilon,
            't1_l2_qs_z':t1_l2_qs_z,
            't1_l1_qs_z_mu':t1_l1_qs_z_mu,
            't1_l1_qs_z_logsigma':torch.exp(t1_l1_qs_z_logsigma),
            't1_l1_qs_z_epsilon':t1_l1_qs_z_epsilon,
            't1_l1_qs_z':t1_l1_qs_z,
            't1_qs_z':t1_qs_z,
            't1_l2_pb_z_mu':t1_l2_pb_z_mu,
            't1_l2_pb_z_logsigma':torch.exp(t1_l2_pb_z_logsigma),
            't1_l1_pb_z_mu':t1_l1_pb_z_mu,
            't1_l1_pb_z_logsigma':torch.exp(t1_l1_pb_z_logsigma),
            't2_l2_t_z_mu':t2_l2_t_z_mu,
            't2_l2_t_z_logsigma':torch.exp(t2_l2_t_z_logsigma),
            't2_l1_t_z_mu':t2_l1_t_z_mu,
            't2_l1_t_z_logsigma':torch.exp(t2_l1_t_z_logsigma),
            #'t2_x_prob':t2_x_prob
            }
        
        return Loss, dict_var

    def rollout(self, sequence, t1, t2):
        self.forward(sequence)
        
        ## at time t1-1, we sample a state z based on belief at time t1-1
        l2_z_mu, l2_z_logsigma = self.l2_b_to_z(self.b[:,t1-1,:])
        l2_z_epsilon = torch.randn_like(l2_z_mu)
        l2_z = l2_z_mu + torch.exp(l2_z_logsigma)*l2_z_epsilon

        l1_z_mu, l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t1-1,:], l2_z), dim = -1))
        l1_z_epsilon = torch.randn_like(l1_z_mu)
        l1_z = l1_z_mu + torch.exp(l1_z_logsigma)*l1_z_epsilon
        current_z = torch.cat((l1_z, l2_z), dim = -1)                        
        
        rollout_x = []
        print_z=[]

        for k in range(t2 - t1 + 1):
            ## predicting states after time t1 using state transition        
            next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(current_z)
            next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
            next_l2_z = next_l2_z_mu + torch.exp(next_l2_z_logsigma)*next_l2_z_epsilon
            
            next_l1_z_mu, next_l1_z_logsigma  = self.l1_transition_z(
                torch.cat((current_z, next_l2_z), dim = -1))
            next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)        
            next_l1_z = next_l1_z_mu + torch.exp(next_l1_z_logsigma)*next_l1_z_epsilon

            next_z = torch.cat((next_l1_z, next_l2_z), dim = -1)

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            mu, logsigma = self.z_to_x(next_z)
            pd = torch.distributions.Normal(mu, torch.exp(logsigma))
            next_x = pd.sample()
            rollout_x.append(next_x)
            print_z.append(next_z)

            current_z = next_z

        rollout_x = torch.stack(rollout_x, dim = 1)
        print_z=torch.stack(print_z,dim=1)
        
        return rollout_x, print_z

    def jumpy_rollout(self, sequence, t1, t2):
        batch_size = self.batch_size
        deltat = np.array([t2 - t1]*batch_size).reshape(batch_size,1)
        deltat = torch.tensor(deltat)
        length = sequence[0].size()[1] - t2
        self.forward(sequence)
        
        ## at time t1-1, we sample a state z based on belief at time t1-1
        l2_z_mu, l2_z_logsigma = self.l2_b_to_z(self.b[:,t1,:])
        l2_z_epsilon = torch.randn_like(l2_z_mu)
        l2_z = l2_z_mu + torch.exp(l2_z_logsigma)*l2_z_epsilon

        l1_z_mu, l1_z_logsigma = self.l1_b_to_z(
            torch.cat((self.b[:,t1,:], l2_z), dim = -1))
        l1_z_epsilon = torch.randn_like(l1_z_mu)
        l1_z = l1_z_mu + torch.exp(l1_z_logsigma)*l1_z_epsilon
        current_z = torch.cat((l1_z, l2_z), dim = -1)                        
        
        rollout_x = []
        
        ## predicting states after time t1 using state transition        
        next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(torch.cat((current_z,deltat),dim=-1))
        next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
        next_l2_z = next_l2_z_mu + torch.exp(next_l2_z_logsigma)*next_l2_z_epsilon
         
        next_l1_z_mu, next_l1_z_logsigma  = self.l1_transition_z(
             torch.cat((current_z, next_l2_z,deltat), dim = -1))
        next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)        
        next_l1_z = next_l1_z_mu + torch.exp(next_l1_z_logsigma)*next_l1_z_epsilon

        next_z = torch.cat((next_l1_z, next_l2_z), dim = -1)

        ## generate an observation x_t1 at time t1 based on sampled state z_t1
        mu, logsigma = self.z_to_x(next_z)
        pd = torch.distributions.Normal(mu, torch.exp(logsigma))
        next_x = pd.sample()
        rollout_x.append(next_x)

        current_z = next_z

        for k in range(length):
            ## predicting states after time t1 using state transition        
            next_l2_z_mu, next_l2_z_logsigma = self.l2_transition_z(torch.cat((current_z,torch.ones(batch_size,1)),dim=-1))
            next_l2_z_epsilon = torch.randn_like(next_l2_z_mu)
            next_l2_z = next_l2_z_mu + torch.exp(next_l2_z_logsigma)*next_l2_z_epsilon
            
            next_l1_z_mu, next_l1_z_logsigma  = self.l1_transition_z(
                torch.cat((current_z, next_l2_z,torch.ones(batch_size,1)), dim = -1))
            next_l1_z_epsilon = torch.randn_like(next_l1_z_mu)        
            next_l1_z = next_l1_z_mu + torch.exp(next_l1_z_logsigma)*next_l1_z_epsilon

            next_z = torch.cat((next_l1_z, next_l2_z), dim = -1)

            ## generate an observation x_t1 at time t1 based on sampled state z_t1
            mu, logsigma = self.z_to_x(next_z)
            pd = torch.distributions.Normal(mu, torch.exp(logsigma))
            next_x = pd.sample()
            rollout_x.append(next_x)
            #print_z.append(next_z)

            current_z = next_z

        rollout_x = torch.stack(rollout_x, dim = 1)
        #print_z=torch.stack(print_z,dim=1)
        
        return rollout_x#, print_z

    def rollout1(self, sequence, t1, t2):
        self.forward(sequence)
        
                                
        
        rollout_x = []

        for z1 in range(0,10000,1000):
            for z2 in range(0,10000,1000):
                z=np.array([[z1,z2,z1,z2]]*200)
                z=torch.from_numpy(z).float()

                ## generate an observation x_t1 at time t1 based on sampled state z_t1
                next_x = self.z_to_x(z)
                rollout_x.append(next_x)

        rollout_x = torch.stack(rollout_x, dim = 1)
        
        return rollout_x