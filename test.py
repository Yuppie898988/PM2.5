import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ans = pd.read_csv(r'.\hw1\ans.csv').iloc[:, 1]
raw_data = pd.read_csv(r'.\hw1\test.csv', names=['id', 'item'] + list(range(9)))
raw_data[raw_data == 'NR'] = 0
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')
theta = pd.read_csv('model_theta.csv').iloc[:, 1]
bias = pd.read_csv('bias.csv').iloc[0, 1]
result = np.zeros((240, 1))
Loss = 0.0
data_row = np.arange(0, raw_data.shape[0], 18)
for row in range(240):
    r = data_row[row]
    data = raw_data.iloc[[r+4, r+5, r+6, r+9, r+12], 2:11]
    result[row] = bias + (data.values.flatten() @ theta)
    Loss += (result[row] - ans[row]) ** 2
plt.plot(np.arange(240), result, label='Prediction')
plt.plot(np.arange(240), ans, label="Truth")
plt.legend()
plt.show()