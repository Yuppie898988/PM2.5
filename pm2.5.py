import pandas as pd
import numpy as np
raw_data = pd.read_csv(r'.\hw1\train.csv', encoding='unicode_escape')
raw_data.drop("´ú¯¸", axis=1, inplace=True)
raw_data.rename(columns={'¤é´Á': "data", "´ú¶µ": "item"}, inplace=True)
raw_data[raw_data == 'NR'] = 0
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')
row_n = raw_data.shape[0]
theta = np.zeros((9 * 18, 1))                                      # 9*18个w
bias = 0
eta = 0.000001
iteration = 10000
y_hat = pd.DataFrame(raw_data.iloc[range(9, row_n, 18), 11:26])        # 存储y_hat，横纵坐标对应之前第9小时的首数据的横纵坐标
y_hat.columns = y_hat.columns.astype(int)
y_hat = y_hat.rename(index={i: i - 9 for i in range(9, row_n, 18)}, columns={i: i - 9 for i in range(9, 24)})
train_data = pd.Series(np.arange(0, 3888, 18))
print(train_data)
for it in range(iteration):
    theta_grad = np.zeros((9 * 18, 1))
    b_grad = 0
    Loss = 0
    for row in train_data:                            # train_data存储训练集的行数，col表示raw_data中input首数据的纵坐标
        for col in range(2, 17):
            b_grad = b_grad - 2 * (y_hat.loc[row, col-2] - bias - (raw_data.iloc[0:18, 2:11].values.flatten() @ theta)[0])
            theta_grad = theta_grad - 2 * (y_hat.loc[row, col-2] - bias -
                                           (raw_data.iloc[0:18, 2:11].values.flatten() @ theta)[0]) * raw_data.iloc[0:18, 2:11].values.reshape(18*9, 1)
            Loss = Loss + (y_hat.loc[row, col-2] - bias - (raw_data.iloc[0:18, 2:11].values.flatten() @ theta)[0]) ** 2
    bias = bias - eta * b_grad
    theta = theta - eta * theta_grad
    print("bias:", bias)
    print("theta:", theta)
    print(Loss/9/18)
print(bias)
print(theta)