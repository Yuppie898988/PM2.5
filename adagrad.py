import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_data = pd.read_csv(r'.\hw1\train.csv', encoding='unicode_escape')
raw_data.drop("´ú¯¸", axis=1, inplace=True)
raw_data.rename(columns={'¤é´Á': "data", "´ú¶µ": "item"}, inplace=True)
raw_data[raw_data == 'NR'] = 0
raw_data = raw_data.apply(pd.to_numeric, errors='ignore')
row_n = raw_data.shape[0]
theta = np.zeros((9 * 5, 1))                                  # 9*5个w
bias = 0
eta = 100
lr_b = 0.0
lr_theta = 0.0
iteration = 10000
y_hat = pd.DataFrame(raw_data.iloc[range(9, row_n, 18), 11:26])        # 存储y_hat，横纵坐标对应之前第9小时的首数据的横纵坐标
y_hat.columns = y_hat.columns.astype(int)
y_hat = y_hat.rename(index={i: i - 9 for i in range(9, row_n, 18)}, columns={i: i - 9 for i in range(9, 24)})
train_data = np.random.choice(np.arange(0, 4320, 18), 192, replace=False)
validation = np.setdiff1d(np.arange(0, 4320, 18), train_data, assume_unique=True)
for it in range(iteration):
    theta_grad = np.zeros((9 * 5, 1))
    b_grad = 0
    Loss = 0
    for row in train_data:                            # train_data存储训练集的行数，col表示raw_data中input首数据的纵坐标
        for col in range(2, 17):
            data = raw_data.iloc[[row+4, row+5, row+6, row+9, row+12], col:col+9]
            b_grad = b_grad - 2.0 * (y_hat.loc[row, col-2] - bias - (data.values.flatten() @ theta)[0])
            theta_grad = theta_grad - 2.0 * (y_hat.loc[row, col-2] - bias - (data.values.flatten() @ theta)[0]) * data.values.reshape(5*9, 1)
            Loss = Loss + (y_hat.loc[row, col-2] - bias - (data.values.flatten() @ theta)[0]) ** 2
    lr_b += b_grad ** 2
    lr_theta += theta_grad ** 2
    bias = bias - (eta / np.sqrt(lr_b)) * b_grad
    theta = theta - (eta / np.sqrt(lr_theta)) * theta_grad
    print("Loss:", Loss/2880)
    if Loss/2880 <= 50:
        print("模型已训练完成")
        break
pd.Series(theta[:, 0]).to_csv('model_theta.csv')
pd.Series(bias).to_csv('bias.csv')
result = np.zeros((720, 1))
true_result = np.zeros((720, 1))
i = 0
for row in validation:
    for col in range(2, 17):
        data = raw_data.iloc[[row+4, row+5, row+6, row+9, row+12], col:col+9]
        result[i] = (data.values.flatten() @ theta) + bias
        true_result[i] = y_hat.loc[row, col-2]
        i += 1
plt.plot(np.arange(720), result, label='Prediction')
plt.plot(np.arange(720), true_result, label="Truth")
plt.legend()
plt.show()
