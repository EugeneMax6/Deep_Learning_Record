# 小批量梯度下降 线性回归
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# random.seed设置固定的随机种子
# 保证每次运行都得到相同数据
np.random.seed(42)

# 得到100个x ，0-2之间
x = 2 * np.random.rand(100, 1)
# y 分布在y=4+3x附近
y = 4 + 3 * x + np.random.randn(100, 1) * 0.5

# plot
plt.scatter(x, y, marker='x', color='red')

# numpy to tensor
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# train dataset
dataset = TensorDataset(x, y)
# random batch data batch_size = 16
# shuffle 随机打乱顺序
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# print len
print('dataloader len = %d' % (int(len(dataloader))))
for index, (data, label) in enumerate(dataloader):
    print('index = %d , num = %d' % (index, len(data)))

# the parameter of Linear model. W for weight, B for bias.
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 50 epochs
for epoch in range(1, 51):
    for batch_idx, (data, label) in enumerate(dataloader):
        # predict data
        h = x * w + b
        # loss,均方误差
        loss = torch.mean((h - y) ** 2)
        loss.backward()     # 计算loss关于w和b的偏导
        # 反方向梯度下降
        w.data -= 0.01 * w.grad.data
        b.data -= 0.01 * b.grad.data
        # 清空w和b的梯度信息
        w.grad.zero_()
        b.grad.zero_()
        # print info
        print('epoch(%d) batch (%d) loss = %.4lf' % (epoch, batch_idx, loss.item()))
# print w and b
print('w = %.4lf, b = %.3lf' % (w.item(), b.item()))
w = w.item()
b = b.item()
x = np.linspace(0, 2, 100)
h = w * x + b
# plot line
plt.plot(x, h)
plt.show()
