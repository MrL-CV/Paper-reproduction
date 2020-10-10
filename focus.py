
import torch
import numpy as np

x=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(x.shape) # shape是(h,w)
h,w=x.shape
x=x.reshape(1,h,w)

x=torch.tensor(x) # 从numpy转换到tensor,转换前后(c,h,w)的位置不变
x=x.cuda().float()
print(x.size())

# tensor中的位置依次是(c,h,w)
# 切片的含义(c,从哪里开始:到哪里结束:步长,从哪里开始:到哪里结束:步长)
# 因为没有batch所以从0轴拼接
x_focus=torch.cat([x[...,0::2,0::2], x[..., 1::2,0::2], x[...,0::2, 1::2], x[..., 1::2, 1::2]], 0)
print(x)
print(x_focus)