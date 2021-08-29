import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
import inception
from PIL import Image
from cache import cache
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import coco
from inception import transfer_values_cache

coco.set_data_dir("/home/ljk/ic/data/coco/")             #设置coco所在的文件夹（路径可以改）
inception.data_dir = 'inception/'    #设置inception模型所在文件夹     

_, filenames_train, captions_train = coco.load_records(train=True)           #用coco包里的加载工具记录训练集图片并将记录保存在一个新的文件中，可以用于快速加载
num_images_train = len(filenames_train)                                                   #训练集图像数量大小

def load_image(path, size=None):                               #定义加载图像函数
    # 使用PIL加载图像
    img = Image.open(path)
#     # 重新定义图像大小
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
#     # 将图像转换到np数组里面
    img = np.array(img)
#     # 将2-dim灰度数组转换成3-dimRGB数组
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img

image_model = inception.Inception()               #加载预训练inception模型
img_size = (224,224)
transfer_values_size = 2048

def print_progress(count, max_count):   #打印处理进度函数
    # 完成的百分比
    pct_complete = count / max_count
    #打印处理进度
    msg = "\r- Progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()
    
def process_images(data_dir, filenames, batch_size=32):       #定义处理图片函数
    # 处理的图片数量
    num_images = len(filenames)
    # 为传输值预分配输出数组，用16位的减少内存占用
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)
    # 初始化文件目录
    start_index = 0
    # 批处理图像文件
    while start_index < num_images:
        #打印处理进度
        print_progress(count=start_index, max_count=num_images)
        # 该批次处理的图片结束目录
        end_index = start_index + batch_size
        # 确保结束索引在范围内
        if end_index > num_images:
            end_index = num_images
        # 加载在该批次的所有图像
        for i, filename in enumerate(filenames[start_index:end_index]):
            j = i + start_index
            # 图像文件路径
            path = os.path.join(data_dir, filename)
            # 返回值是一堆数组
            img = load_image(path, size=img_size)
            # in处理
            tv = image_model.transfer_values(image = img)
            #存数据
            transfer_values[j,:] = tv
        # 增加下一个循环迭代的索引
        start_index = end_index
    # 打印换行符.
    print()
    return transfer_values

def process_images_train():        #定义处理训练集图片的函数
    print("Processing {0} images in training-set ...".format(len(filenames_train)))
    # 缓存文件的路径
    cache_path = os.path.join(coco.data_dir, 'inception_coco_train.pkl')
    #如果缓存存在的话我们就直接加载它，这样可以更快速的进入模型
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            data_dir=coco.train_dir,
                            filenames=filenames_train)
    return transfer_values

transfer_values_train = process_images_train()    #取出所有图片的传输值

mark_start = 'ssss '    #开头标志（取一个最不可能出现在英文中的单词作为标记词）
mark_end = ' eeee'    #结束标志（同上）


def mark_captions(captions_listlist):   #定义一个函数，将上面两个标记词包装一下，然后加到所有人工描述的句子上面去
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                       for captions_list in captions_listlist]

    return captions_marked

captions_train_marked = mark_captions(captions_train)   #运行上面定义那个函数


def flatten(captions_listlist):         #定义一个函数，将上面加到人工描述的句子拼合起来
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]

    return captions_list

captions_train_flat = flatten(captions_train_marked)    #运行上面那个函数
num_words = 10000    #设置词汇表中的最大单词数，意味着只会使用训练集中的描述中最频繁出现的10000个单词（设置词汇数量越高越难训练）


class TokenizerWrap(Tokenizer):        #将需要的所有函数打包起来一块使用

    def __init__(self, texts, num_words=None):

        Tokenizer.__init__(self, num_words=num_words)

        # 从文本中创建词汇表
        self.fit_on_texts(texts)

        # 创建从整数标记到单词的反向查找
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):           #从整数标记中查找单词

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):              #将整数标记转换为字符串

        # 创建单个单词的列表
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # 将单词连接到单个字符串，所有单词之间有空格
        text = " ".join(words)

        return text

    def captions_to_tokens(self, captions_listlist):             #将带有文本描述的列表转换为整数标记列表
        #采用文本列表
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]

        return tokens
#获取开始标记的整数标记；获取结束标记的整数标记；将训练集中的所有描述转换为整数标记序列，我们得到了一个列表中的列表。
tokenizer = TokenizerWrap(texts=captions_train_flat,num_words=num_words)
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]
tokens_train = tokenizer.captions_to_tokens(captions_train_marked)


def get_random_caption_tokens(idx):               #给定训练集中图像的索引列表，为随机描述选择标记序列，并返回所有这些标记序列的列表。
    
    # 为结果初始化一个空白的列表
    result = []

    # 对于每一个指数
    for i in idx:
        # 索引i指向训练集中的图像。 训练集中的每个图像至少有5个标题已经转换为tokens_train中的标记。我们希望随机选择其中一个标记序列。
        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]
        result.append(tokens)

    return result


def batch_generator(batch_size):          #用于创建随机批次的训练数据的生成器。

    # 无限循环.
    while True:
        # 获取训练集中图像的随机索引列表。
        idx = np.random.randint(num_images_train,
                                size=batch_size)

        # 获取这些图像的预先计算的传递值，这些是预训练图像模型的输出。
        transfer_values = transfer_values_train[idx]
        tokens = get_random_caption_tokens(idx)

        # 计算所有这些标记序列中的标记数。
        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)

        # 用零填充所有其他标记序列，使它们都具有相同的长度，并且可以作为numpy数组输入到神经网络。
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        # 进一步准备标记序列。神经网络的解码器部分将尝试将标记序列映射到自身移位一个时间步长。
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        # 输入数据的字典。 因为有几个输入，所以使用命名的dict来确保正确分配数据。
        x_data = \
            {
                'decoder_input': decoder_input_data,
                'transfer_values_input': transfer_values
            }

        # 输出数据的字典
        y_data = \
            {
                'decoder_output': decoder_output_data
            }

        yield (x_data, y_data)

batch_size = 1450           #设置训练期间使用的批量大小
generator = batch_generator(batch_size=batch_size)        #创建数据生成器的一个例子
batch = next(generator)          #通过创建一批数据来测试数据生成器

num_captions_train = [len(captions) for captions in captions_train]    #这是训练集中每个图像的描述数量。
total_num_captions_train = np.sum(num_captions_train)     #这是训练集中的描述总数。
steps_per_epoch = int(total_num_captions_train / batch_size)     #总的批次量

state_size = 512   #解码器由3个GRU组成，其内部大小为512
embedding_size = 128    #嵌入层的大小为128
transfer_values_input = Input(shape=(transfer_values_size,),        
                              name='transfer_values_input')         #将传输值输入到解码器中
decoder_transfer_map = Dense(state_size,                 
                             activation='tanh',
                             name='decoder_transfer_map')               #使用完全连接的层来映射4096到512个元素的向量。并使用tanh激活函数来限制-1，1之间的输出
decoder_input = Input(shape=(None, ), name='decoder_input')      #这是标记序列到解码器的输入。
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')          #将整数标记的序列转换为向量序列。
#创建解码器的3个GRU层
decoder_gru1 = GRU(state_size, name='decoder_gru1',       
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')          #GRU层输出形状为[batch_size，sequence_length，state_size]的张量，其中每个“字”被编码为长度为state_size（512）的向量。 需要将其转换为整数标记序列，可以将其解释为词汇表中的单词。

#将解码器的所有层连接到传输值的输入。
def connect_decoder(transfer_values):
    # 映射传输值，使维度与GRU层的内部状态匹配。 可以使用映射的传输值作为GRU层的初始状态。
    initial_state = decoder_transfer_map(transfer_values)

    # 使用输入层启动解码器网络。
    net = decoder_input

    # 连接嵌入层
    net = decoder_embedding(net)

    # 连接所有GRU层
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # 连接转换为独热编码数组的最终全连接层
    decoder_output = decoder_dense(net)

    return decoder_output

decoder_output = connect_decoder(transfer_values=transfer_values_input)             #连接并创建用于训练的模型。 输入传递值和整数标记序列，并输出可以转换为整数令牌的独热编码阵列的序列。
decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])


def sparse_cross_entropy(y_true, y_pred):       #计算y_true和y_pred之间的交叉熵损失。

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    # 将输出的指标减少到平均函数值
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

optimizer = RMSprop(lr=0.5e-4)    #使用RMSprop优化器
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))     #为解码器的输出创建一个占位符变量

#编译训练模型
decoder_output = connect_decoder(transfer_values=transfer_values_input)
decoder_model = Model(inputs=[transfer_values_input, decoder_input],outputs=[decoder_output])
decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])

#编写检查点回调
path_checkpoint = 'inception_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)
callback_tensorboard = TensorBoard(log_dir='./22_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]
path_checkpoint = 'inception_checkpoint.keras'
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

#开始训练
# tol_epochs = 30
# while i < tol_epochs :
# 	i = 3
decoder_model.fit_generator(generator=generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=50,
                            callbacks=callbacks)