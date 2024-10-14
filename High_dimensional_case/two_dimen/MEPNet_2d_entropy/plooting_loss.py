import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
# Load the data from the provided files
#file_noentropy_path = 'training_records_2d_noentropy.txt'
file_entropy_path = 'training_records_2d_entropy.txt'

#data_noentropy = pd.read_csv(file_noentropy_path, delim_whitespace=True)
data_entropy = pd.read_csv(file_entropy_path, delim_whitespace=True)

# Extract the third columns from each dataframe
#third_col_noentropy = data_noentropy.iloc[:, 2]
third_col_entropy = data_entropy.iloc[:, 2]

# Plot the third columns on the same graph
plt.figure(figsize=(10, 6))
#plt.plot(range(len(third_col_noentropy))[0:100000], np.log(third_col_noentropy[0:100000]), label='No Entropy')
plt.plot(range(len(third_col_entropy))[0:100000], np.log(third_col_entropy[0:100000]))

#plt.plot(range(len(third_col_noentropy)), np.log(third_col_noentropy), label='No Entropy')
#plt.plot(range(len(third_col_entropy)), np.log(third_col_entropy), label='Entropy')
plt.xlabel('Epoch')
plt.ylabel('Log(MSE)')
#plt.yscale('log', base=10)
#plt.ylim(1*1e-3, 8e-3)  # 设置缩放范围
#plt.ylim(-7, -5)  # 设置缩放范围
plt.legend()
plt.grid(True)
plt.show()
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the provided file
file_entropy_path = 'training_records_2d_entropy.txt'
data_entropy = pd.read_csv(file_entropy_path, delim_whitespace=True)

# Extract the third column from the dataframe
third_col_entropy = data_entropy.iloc[:, 2]

# Set up the plot with larger figure size
plt.figure(figsize=(12, 8))

# Plot the third column with a thicker, blue line
plt.plot(range(len(third_col_entropy))[0:100000], np.log(third_col_entropy[0:100000]), 
         color='blue', linewidth=3)

# Set labels with larger, bold font
plt.xlabel('Epoch', fontsize=30, fontweight='bold')
plt.ylabel('Log(MSE)', fontsize=30, fontweight='bold')

# Customize the legend
#plt.legend(['Entropy'], fontsize=20, loc='best')

# Add grid and customize ticks
plt.grid(True, linestyle='--', alpha=1.0)
plt.tick_params(axis='both', which='major', labelsize=30)

# Set title with larger, bold font
#plt.title('Loss Curve', fontsize=16, fontweight='bold')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
