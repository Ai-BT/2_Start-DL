# %%

import torch

# 1. expand 

x = torch.FloatTensor([[[1,2]],[[3,4]]])
print(x)
print(x.size())

y = x.expand(* [2,3,2]) 
print(y)
print(y.size())

# %%

# 2 randperm

x = torch.randperm(10)
print(x)
print(x.size())

# %%

# 3 argmax
# index 리턴

# torch.randperm(3**3) -> 27 까지의 최대값
# -1 은 자동으로 합이 맞춰지는 형태

x = torch.randperm(3**3).reshape(3, 3, -1)

print(x)
print(x.size())

# argmax
# 차원에서 가장 큰 위치에 있는 것들을 추출
y = x.argmax(dim=-1)
print(y)

# %%

# 4. dim

# dim : 어느 차원을 늘릴 것인지 인자로 줄 수 있음, dim=0 : 첫번째 차원을 늘림

x = torch.FloatTensor([[1, 2], [3, 4], [9, 10]]) # (3,2)
y = torch.FloatTensor([[5, 6], [7, 8]]) # (2,2)
z = torch.FloatTensor([[5, 6, 7], [8, 9, 10]]) # (3,2)


print(torch.cat([x, y], dim=0)) # 0 번째 차원
print(torch.cat([y, z], dim=1)) # 1 번째 차원
# print(torch.cat([x, y], dim= -1))


batch_size, N, K = 3, 10, 256

x = torch.rand(batch_size, N, K) # [M, N, K]
y = torch.rand(batch_size, N, K) # [M, N, K]


output1 = torch.cat([x,y], dim=1) #[M, N+N, K]
output2 = torch.cat([x,y], dim=2) #[M, N, K+K]
output3 = torch.cat([x,y], dim=0) #[M+M, N, K]

print(output3.size())


# %%

# 5. top-k
# value 와 index 리턴

# x = (3,3,3)
# k=1 은 탑 one 만 뽑는다
# dim = -1 마지막 차원
values, indices = torch.topk(x, k=1, dim=-1) # value, index 리턴

print(values.size())
print(indices.size())

# squeeze 차원 축소
print(values.squeeze(-1))
print(indices.squeeze(-1))


# %%

# 6. masked_fill

x = torch.FloatTensor([ i for i in range(3**2)]).reshape(3,-1)

print(x)

mask = x > 4
print(mask)

y = x.masked_fill(mask, value=-1) # mask 가 true 이면 거기를 -1 채워
print(y)

# %%
