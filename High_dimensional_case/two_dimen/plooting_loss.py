import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the provided files
file_noentropy_path = 'training_records_2d_noentropy.txt'
file_entropy_path = 'training_records_2d_entropy.txt'

data_noentropy = pd.read_csv(file_noentropy_path, delim_whitespace=True)
data_entropy = pd.read_csv(file_entropy_path, delim_whitespace=True)

# Extract the third columns from each dataframe
third_col_noentropy = data_noentropy.iloc[:, 2]
third_col_entropy = data_entropy.iloc[:, 2]

# Plot the third columns on the same graph
plt.figure(figsize=(10, 6))
plt.plot(range(len(third_col_noentropy))[0:100000], np.log(third_col_noentropy[0:100000]), label='No Entropy')
plt.plot(range(len(third_col_entropy))[0:100000], np.log(third_col_entropy[0:100000]), label='Entropy')

#plt.plot(range(len(third_col_noentropy)), np.log(third_col_noentropy), label='No Entropy')
#plt.plot(range(len(third_col_entropy)), np.log(third_col_entropy), label='Entropy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.yscale('log', base=10)
#plt.ylim(1*1e-3, 8e-3)  # 设置缩放范围
#plt.ylim(-7, -5)  # 设置缩放范围
plt.legend()
plt.grid(True)
plt.show()
