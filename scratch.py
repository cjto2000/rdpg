import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# x = torch.FloatTensor([[1, 2], [2, 3]])
# print(x.shape)
# # x = x.view(1, -1)
# x = x.unsqueeze(0)
# print(x.shape)
# fc1 = nn.Linear(2, 10)
# print(fc1(x))

x = torch.FloatTensor([[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]]])
y = x[:, :-1]
print(x.shape)
print(y.shape)
print(x)
print(y)

