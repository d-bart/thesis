import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4,5,figsize=(25,20))
i=0
for belief_state_size in [10,50,100,200]:
    k=0
    for state_size in [2,4,8,10,15]:
        validation_loss = []
        training_loss = []
        with open("./log/loginfo_new_belief_{}_state{}_clock.txt".format(belief_state_size,state_size), 'r') as file_handle:
            for line in file_handle:
                line = line.strip()
                field = line.split()
                if field[-1] != "nan":
                    validation_loss.append(float(field[-1]))
                if field[-3] != "nan":
                    training_loss.append(float(field[-3][:-1]))
            
        axs[i,k].plot(validation_loss, label='Validation Loss')
        axs[i,k].plot(training_loss,color='red', label='Training Loss')
        axs[i,k].set_title('Belief = {}, state = {}'.format(belief_state_size,state_size))
        k=k+1
    i=i+1
        
axs[0,4].legend(loc='upper right',bbox_to_anchor=(1.4, 1))


