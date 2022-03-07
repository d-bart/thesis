import numpy as np
import matplotlib.pyplot as plt

validation_loss = []
training_loss = []
with open('./log/loginfo_new_decoder_dblock_clock_A_random.txt', 'r') as file_handle:
    for line in file_handle:
        line = line.strip()
        field = line.split()
        if field[-1] != "nan":
            validation_loss.append(float(field[-1]))
        if field[-3] != "nan":
            training_loss.append(float(field[-3][:-1]))
            
plt.plot(validation_loss, label='Validation Loss')
plt.plot(training_loss,color='red', label='Training Loss')
plt.legend(loc='upper right',bbox_to_anchor=(1.4, 1))
plt.show()
