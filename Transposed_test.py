# 转置卷积的测试
# 修改基于https://github.com/naokishibuya/deep-learning/blob/master/python/transposed_convolution.ipynb

import numpy as np
import matplotlib.pyplot as plt

# import keras
# import keras.backend as K
# from keras.layers import Conv2D
# from keras.models import Sequential

# 改成列矩阵
def col_matrix(m):
    return m.flatten().reshape(-1, 1)

# 求矩阵最大最小值
def maxandmin(m):
    max = m.max()
    min = m.min()

    return max, min

# 卷积矩阵展开
def convolution_matrix(m, k):

    m_rows, m_cols = len(m), len(m[0])  # matrix rows, cols
    k_rows, k_cols = len(k), len(k[0])  # kernel rows, cols

    # output matrix rows and cols
    rows = m_rows - k_rows + 1
    cols = m_rows - k_rows + 1

    # print(m)
    # print(k)

    # convolution matrix
    v = np.zeros((rows*cols, m_rows, m_cols))

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            v[i][r:r+k_rows, c:c+k_cols] = k
    # 相当于flatten()
    v_reshape = v.reshape((rows*cols), -1)
    return v, v_reshape

# 可视化矩阵
def show_matrix(m, color, cmap, title=None, vmin=None, vmax=None):
    rows, cols = len(m), len(m[0])
    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_yticks(list(range(rows)))
    ax.set_xticks(list(range(cols)))
    ax.xaxis.tick_top()
    # 标题选项
    if title is not None:
        ax.set_title('{} {}'.format(title, m.shape), y=-0.5/rows)
    plt.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()

    # 每行每列打印矩阵数字
    for r in range(rows):
        for c in range(cols):
            text = '{:>3}'.format(int(m[r][c]))
            ax.text(c-0.2, r+0.15, text, color=color, fontsize=15)
    plt.show()

# 卷积操作
def conv_3(input, kernel):
        # 输入特征图和kernel的size
    row_in = len(input)
    col_in = len(input[0])
    row_k = len(kernel)
    col_k = len(kernel[0])

    # 输出特征图的size
    row_o = row_in-row_k+1
    col_o = col_in-col_k+1

    # print(input)
    # print(kernel)

    output = np.zeros((row_o, col_o), dtype=input.dtype)

    for i in range(row_o):
        for j in range(col_o):
            output[i][j] = np.sum(input[i:i+row_k, j:j+col_k]*kernel)

    return output

# 打印输入矩阵使用的是蓝色
def show_inputs(m, title='Inputs', vmin=None, vmax=None):
    show_matrix(m, 'b', plt.cm.Blues, title, vmin, vmax)


def show_kernel(m, title='Kernel', vmin=None, vmax=None):
    show_matrix(m, 'r', plt.cm.Purples, title, vmin, vmax)


def show_output(m, title='Output', vmin=None, vmax=None):
    show_matrix(m, 'g', plt.cm.GnBu, title, vmin, vmax)


# 随机矩阵
input = np.random.randint(1, 9, size=(4, 4))
# 显示随机矩阵的深度图
max, min = maxandmin(input)
# show_inputs(input,vmin=max,vmax=min)

# 创建卷积所用的kernel
kernel = np.random.randint(1, 5, size=(3, 3))
max, min = maxandmin(kernel)
# show_kernel(kernel,vmin=max,vmax=min)

# 输出矩阵
output=conv_3(input,kernel)
# print(output)
max,min=maxandmin(output)
# show_output(output,vmin=max,vmax=min)

# 卷积矩阵
conv_m, conv_m_reshape = convolution_matrix(input, kernel)
# print(conv_m)
# print(conv_m_reshape)

input_col = col_matrix(input)
# print(input_col)

# 使用卷积矩阵和输入矩阵的列形式进行矩阵乘法(@是矩阵乘法)
# [4,4,4]->[4,16]与[4,4]->[16,1]相乘
# 最后得到[4,1]->[2,2]
output_conv_m_reshape_input_col = conv_m_reshape@input_col
# print(output)
print(input)
print(output_conv_m_reshape_input_col.reshape(2,-1))

# 相反,可以通过
# [4,4,4]->[4,16]->[16,4]与[2,2]->[4,1]相乘
# 最后得到[16,1]->[4,4]
# 得到上采样效果
# 但是有一点,反卷积的kernel_size比被卷积特征图的size要大
input_2=np.random.randint(1, 9, size=(2, 2))
input_2_col=col_matrix(input_2)
output_conv_m_reshape_input2_col=conv_m_reshape.T@input_2_col
print(input_2)
print(output_conv_m_reshape_input2_col.reshape(4,-1))
# -----------------------------------------------------------------------------
# 总结:转置卷积比上采样更加高效而且不需要通过补零来完成