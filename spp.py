# SPP实现
# 修改基于https://www.cnblogs.com/marsggbo/p/8572846.html

import math
import torch
import torch.nn.functional as F
import cv2

# 构建SPP层
# 用法spp=SPPLayer(num_levels, pool_type='max_pool'),就完成了初始化
# 继续使用:spp.forward(x),就可以使用类内函数


class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__() # 用于继承

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):  # x是需要进行spp池化的特征图
        # num, c, h, w = x.size() # num:batch c:通道数 h:高 w:宽
        x=torch.tensor(x) # 从numpy转换到tensor
        h,w,c=x.size()
        x=x.view(c,h,w)
        x=x.cuda().float()

        num = 1

        for i in range(self.num_levels):

            # 精华部分
            level = i+1  # 最后输出的形状,1*1,2*2...4*4
            print("当前层级",level)
            kernel_size = (math.ceil(h / level), math.ceil(w / level))  # 池化核尺寸
            print("当前核尺寸",kernel_size)
            stride = (math.ceil(h / level), math.ceil(w / level))  # 池化步长
            print("当前步长大小",stride)
            pooling = (math.ceil((kernel_size[0]*level-h+1)/2), math.ceil((kernel_size[1]*level-w+1)/2))  # 边界补齐宽度
            print("当前补齐大小",pooling)

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
            else:  # 否则就是平均池化
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)

            print("当前层级拉平前的尺寸为:",tensor.size())
            print("当前层级拉平后的尺寸为:",tensor.view(num,-1).size())
            print("\n")

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)  # 将特征图变成(batch,*)的形状
            else:
                # 后面的向量按照*轴进行拼接,返回拼接的结果
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


x = cv2.imread('bus.jpg') # 导入图片或者特征图均可

spp = SPPLayer(4, pool_type='max_pool') # 初始化
x_spp = spp.forward(x) # 处理数据

print("空间金字塔池化后大小",x_spp.size())
print("\n")
