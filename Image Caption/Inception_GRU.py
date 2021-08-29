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
#GRU层输出形状为[batch_size，sequence_length，state_size]的张量，
#其中每个“字”被编码为长度为state_size（512）的向量。 
#需要将其转换为整数标记序列，可以将其解释为词汇表中的单词。
decoder_gru1 = GRU(state_size, name='decoder_gru1',       
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')          

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

#编译训练模型
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
decoder_output = connect_decoder(transfer_values=transfer_values_input)
decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

#编写检查点回调
path_checkpoint = 'inception_checkpoint.keras'
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

#生成图像描述
def generate_caption(image_path, max_tokens=20):     #为给定路径中的图像生成描述
    imag = load_image(image_path, size=img_size)
    image_batch= image_model.transfer_values(image = imag)   #获得输入图片的传输值
    transfer_values = np.expand_dims(image_batch, axis=0)           #将3维度数组扩充到4维度去
    #预先分配用作解码器输入的2-dim阵列。这只包含一个整数标记序列，但解码器模型需要一批序列。
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)
    #第一个输入的标记应该是开始标记“ssss”
    token_start = tokenizer.word_index[mark_start.strip()]
    token_int = token_start
    #用空白去初始化输出句子
    output_text = ''
    #初始化我们已经处理的标记数量
    count_tokens = 0
    while token_int != token_end and count_tokens < max_tokens:                            #开始不断迭代直到输出结束标志“eeee”
        # 使用已采样的最后一个标记将输入序列更新到解码器。在第一次迭代中，这将把第一个元素设置为start-token。       
        decoder_input_data[0, count_tokens] = token_int
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