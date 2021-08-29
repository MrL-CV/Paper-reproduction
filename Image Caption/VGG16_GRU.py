import matplotlib
matplotlib.use('Agg')                 #为了解决不能在linux上画图的问题
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
import coco
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

coco.set_data_dir("/home/ljk/ic/data/coco/")                                               ####记得改路径
_, filenames_train, captions_train = coco.load_records(train=True)
image_model = VGG16(include_top=True, weights='imagenet')

def load_image(path, size=None):                               #定义加载图像函数
    # 使用PIL加载图像
    img = Image.open(path)
    # 重新定义图像大小
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)
    # 将图像转换到np数组里面
    img = np.array(img)
    # 归一化像素
    img = img / 255.0
    # 将2-dim灰度数组转换成3-dimRGB数组
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    return img

transfer_layer = image_model.get_layer('fc2')         #将VGG16的最后一层fc2层去掉，替换成自己创建的传输层
image_model_transfer = Model(inputs=image_model.input,                
                             outputs=transfer_layer.output)                                          #重新创建一个模型，这个模型没有最后fc2层，但是有传输层
img_size = K.int_shape(image_model.input)[1:3]                     #定义输入图像的大小
transfer_values_size = K.int_shape(transfer_layer.output)[1]        #定义输出的传输值向量的大小

mark_start = 'ssss '               #开头标志（取一个最不可能出现在英文中的单词作为标记词）
mark_end = ' eeee'

def mark_captions(captions_listlist):                #定义一个函数，将上面两个标记词包装一下，然后加到所有人工描述的句子上面去
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    return captions_marked

def flatten(captions_listlist):              #定义一个函数，将上面加到人工描述的句子拼合起来
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    return captions_list

captions_train_marked = mark_captions(captions_train)
captions_train_flat = flatten(captions_train_marked)
num_words = 10000               #设置词汇表中的最大单词数，意味着只会使用训练集中的描述中最频繁出现的10000个单词（设置词汇数量越高越难训练）

class TokenizerWrap(Tokenizer):             #将需要的所有函数打包起来一块使用
    def __init__(self, texts, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))
    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word 
    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]
        text = " ".join(words)
        return text
    def captions_to_tokens(self, captions_listlist):
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        return tokens

#获取开始标记的整数标记；获取结束标记的整数标记；将训练集中的所有描述转换为整数标记序列，我们得到了一个列表中的列表。
tokenizer = TokenizerWrap(texts=captions_train_flat,num_words=num_words)
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]

state_size = 512
embedding_size = 128
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')
decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

#定义RNN的3个GRU层
decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(transfer_values):                     #将两个模型连接起来 
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)
    return decoder_output

#定义优化器等，编译模型
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
decoder_output = connect_decoder(transfer_values=transfer_values_input)
decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

#记录训练的还原点
path_checkpoint = '22_checkpoint.keras'
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
    
def generate_caption(image_path, max_tokens=20):     #为给定路径中的图像生成描述
    image = load_image(image_path, size=img_size)
    image_batch = np.expand_dims(image, axis=0)    #将3维度数组扩充到4维度去
    transfer_values = image_model_transfer.predict(image_batch)  #获取图片传输值
    #预先分配用作解码器输入的2-dim阵列。这只包含一个整数标记序列，但解码器模型需要一批序列。
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    print(decoder_input_data)
    print("\n")
    #第一个输入的标记应该是开始标记“ssss”
    token_start = tokenizer.word_index[mark_start.strip()]
    token_int = token_start
    #用空白去初始化输出句子
    output_text = ''
    #初始化我们已经处理的标记数量
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:     #开始不断迭代直到输出结束标志“eeee”
        # 使用已采样的最后一个标记将输入序列更新到解码器。在第一次迭代中，这将把第一个元素设置为start-token。       
        decoder_input_data[0, count_tokens] = token_int
        print(decoder_input_data)
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }
        # 在调用predict（）时返回GRU状态，然后在下次调用predict（）时提供这些GRU状态，减少计算量
        decoder_output = decoder_model.predict(x_data)
        # 获取最后一个预测的标记作为单热编码数组
        token_onehot = decoder_output[0, count_tokens, :]
        # 转换为整数标记
        token_int = np.argmax(token_onehot)
        # 查找与此整数标记对应的单词
        sampled_word = tokenizer.token_to_word(token_int)
        output_text += " " + sampled_word
        # 增加标记计数器
        count_tokens += 1
    # 这是解码器输出的标记序列(最终迭代之后)。
    output_tokens = decoder_input_data[0]
    print("\n")
    print(output_text[:-5] + ".")
    
path_1 = "/home/ljk/ic/data/test/"                      #测试图片路径
my_files = os.listdir(path_1)
my_files.sort()

while True:                       #总程序
    content = input('请输入第几张图片（退出输入qq）：')
    if content == 'qq':
        break
    else:
        listFiles = []
        for files in my_files:
            listFiles.append(files)
        content_1 = int(content) - 1
        path_2 = "/home/ljk/ic/data/test/"+listFiles[ content_1 ]
        print(listFiles[ content_1 ])
        generate_caption(path_2)
        print("\n")