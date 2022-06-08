# %%

# Operations

import torch

# 1. 사칙연산
a = torch.FloatTensor([[1,2],[3,4]])
b = torch.FloatTensor([[2,2],[3,3]])

print(a + b)
print(a - b)
print(a / b)
print(a * b)
print(a ** b)
print(a == b)

# %%

# 2. inplace Operation
# 새로운 메모리를 사용하지 않고, 기존에 메모리에 덮어쓰우는 방법

print(a)
print(a.mul(b))
print(a)
print(a.mul_(b))
print(a)

# %%

# 3. sum, mean, dim

# sum, mean

x = torch.FloatTensor([[1,2],[3,4]])

print(x.sum())
print(x.mean())


# 특정 dim 에 대해서 sum을 하고 싶다
# 쉽게 이해하면 없어지는 dim 이라고 생각하면 된다.
# 팍 찝어서 합쳐지는거

print(x.sum(dim=0))
print(x.sum(dim=-1))



# %%

# 4. Broadcast 기능

# Tensor + Scalar
x = torch.FloatTensor([[1,2],[3,4]])
y = 1

print(x.size())
z = x + y
print(z)
print(z.size())

# Tensor + Vector
x = torch.FloatTensor([[1,2], [4,8]])
y = torch.FloatTensor([3,5])

z = x + y
print(z)

# Tensor + Tensor
x = torch.FloatTensor([[1,2]])
y = torch.FloatTensor([[3],[5]])

z = x + y
print(z)

# %%
