import math

import torch
import torch.nn as nn

a = torch.Tensor([[3,4,5],[1,2,3]])
linear_a = nn.Linear(3,64)
b = torch.Tensor([[2,4,6],[3,6,9]])
linear_b = nn.Linear(3,64)
print(a)
print(a.size(1)) # 3
# print(linear_(a).shape) # 2x3 * 3x64 = [2,64]
# print(linear_(a).view(64, 2, -1, 1).transpose(1,2).shape)
scores_a = linear_a(a).view(64, 2, -1, 1).transpose(1,2)
scores_b = linear_b(b).view(64, 2, -1, 1).transpose(1,2)
#
# scores_a = scores_a / math.sqrt(5)
# scores = torch.matmul(scores_a, scores_b.transpose(2,3))
# print(scores.shape) # shape (64,1,2,2)
# scores = scores.masked_fill
# test = torch.arange(0, 64).unsqueeze(1)
# print(test)

encoding_ = torch.zeros(200000, 10)
position = torch.arange(10, 200000).unsqueeze(1)
print(encoding_[:,0::2])