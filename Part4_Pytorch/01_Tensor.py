# Tensor 는 3차원 이상의 형상을 말한다.
# GPU 사용하는 이유는 병렬로 계산하기 위해서
# k 개의 샘플을 뽑아서, mini batch를 해서 계산하다.

# %%
import torch

# 1. float_tensor
float_tensor = torch.FloatTensor([ [1,2],[3,4] ])
float_tensor

# %%

# 2. 3,2 의 가비지 값 생성
x = torch.FloatTensor(3, 2)
x

# %%

# 3. Numpy Compatiblity

import numpy as np

y = np.array([[1,2],[3,4]])

y = torch.from_numpy(y)
print(y, type(y))

y = y.numpy()
print(y, type(y))


# %%

x = torch.FloatTensor(3,2,2)

# 사이즈
print(x.size())
print(x.shape)

# 차원
print(x.dim())

# %%
