# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

def dataPreHandle(X,Y):
    '''
        数据预处理
    '''
    resultList = []
    # 初始化列向量的维度
    colDimension = X[0].shape[1]

    for xItem,yItem in zip(X,Y):
        # print(xItem.shape,yItem.shape)  #(886, 39) (886, 24)
        xReshape = Standardize(xItem).reshape((1,-1,colDimension)).astype('float32')
        # print(type(xReshape),xReshape.shape)   #(1, 886, 39)
        yReshape = Standardize(yItem).astype('float32')
        # print(type(yReshape),yReshape.shape)   #(886, 24)
        resultList.append([xReshape,yReshape])
        # print(len(resultList[0]))   #2
        # break

    return resultList


def Standardize(sentenceMatrix):
    '''
        矩阵标准化处理

    '''
    # 矩阵减去均值（中心化操作）
    centerized = sentenceMatrix - np.mean(sentenceMatrix,axis=0)
    # 除以标准差
    normalized = centerized/np.std(centerized,axis=0)

    return normalized


class LSTMcell(object):
  def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):

      # var
      # incoming是用来接收输入数据的，其形状为[n_samples, n_steps, D_cell]
      self.incoming = incoming
      # 输入的维度
      self.D_input = D_input
      # LSTM的hidden state的维度，同时也是memory cell的维度
      self.D_cell = D_cell
      # parameters
        # 输入门的 三个参数
        # igate = W_xi.* x + W_hi.* h + b_i
      self.W_xi = initializer([self.D_input, self.D_cell])
      self.W_hi = initializer([self.D_cell, self.D_cell])
      self.b_i  = tf.Variable(tf.zeros([self.D_cell]))
        # 遗忘门的 三个参数
        # fgate = W_xf.* x + W_hf.* h + b_f
      self.W_xf = initializer([self.D_input, self.D_cell])
      self.W_hf = initializer([self.D_cell, self.D_cell])
      self.b_f  = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # 输出门的 三个参数
        # ogate = W_xo.* x + W_ho.* h + b_o
      self.W_xo = initializer([self.D_input, self.D_cell])
      self.W_ho = initializer([self.D_cell, self.D_cell])
      self.b_o  = tf.Variable(tf.zeros([self.D_cell]))
        # 计算新信息的三个参数
        # cell = W_xc.* x + W_hc.* h + b_c
      self.W_xc = initializer([self.D_input, self.D_cell])
      self.W_hc = initializer([self.D_cell, self.D_cell])
      self.b_c  = tf.Variable(tf.zeros([self.D_cell]))

      # 最初时的hidden state和memory cell的值，二者的形状都是[n_samples, D_cell]
      # 如果没有特殊指定，这里直接设成全部为0
      init_for_both = tf.matmul(self.incoming[:,0,:], tf.zeros([self.D_input, self.D_cell]))
      self.hid_init = init_for_both
      self.cell_init = init_for_both
      # 所以要将hidden state和memory并在一起。
      self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
      # 需要将数据由[n_samples, n_steps, D_cell]的形状变成[n_steps, n_samples, D_cell]的形状
      self.incoming = tf.transpose(self.incoming, perm=[1,0,2])


  def one_step(self, previous_h_c_tuple, current_x):

      # 再将hidden state和memory cell拆分开
      prev_h, prev_c = tf.unstack(previous_h_c_tuple)
      # 这时，current_x是当前的输入，
      # prev_h是上一个时刻的hidden state
      # prev_c是上一个时刻的memory cell

      # 计算输入门
      i = tf.sigmoid(
          tf.matmul(current_x, self.W_xi) +
          tf.matmul(prev_h, self.W_hi) +
          self.b_i)
      # 计算遗忘门
      f = tf.sigmoid(
          tf.matmul(current_x, self.W_xf) +
          tf.matmul(prev_h, self.W_hf) +
          self.b_f)
      # 计算输出门
      o = tf.sigmoid(
          tf.matmul(current_x, self.W_xo) +
          tf.matmul(prev_h, self.W_ho) +
          self.b_o)
      # 计算新的数据来源
      c = tf.tanh(
          tf.matmul(current_x, self.W_xc) +
          tf.matmul(prev_h, self.W_hc) +
          self.b_c)
      # 计算当前时刻的memory cell
      current_c = f*prev_c + i*c
      # 计算当前时刻的hidden state
      current_h = o*tf.tanh(current_c)
      # 再次将当前的hidden state和memory cell并在一起返回
      return tf.stack([current_h, current_c])


  def all_steps(self):
      # 输出形状 : [n_steps, n_sample, D_cell]
      hstates = tf.scan(fn = self.one_step,
                        elems = self.incoming, #形状为[n_steps, n_sample, D_input]
                        initializer = self.previous_h_c_tuple,
                        name = 'hstates')[:,0,:,:]
      return hstates


# 正交矩阵初始化
# 正交矩阵初始化是有利于gated_rnn的学习的方法
def orthogonal_initializer(shape,scale = 1.0):
  #https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
  scale = 1.0
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape) #this needs to be corrected to float32
  return tf.Variable(scale * q[:shape[0], :shape[1]],trainable=True, dtype=tf.float32)


def weight_init(shape):
    '''
        这里的shape是一个list
    '''
    # 产生一个均匀分布，形状为shape的tensor
    initial = tf.random_uniform(shape,minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]), maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
    return tf.Variable(initial,trainable=True)


def bias_init(shape):
    '''
        shapes是一个list
    '''
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# 洗牌
def shufflelists(data):
  ri=np.random.permutation(len(data))
  data=[data[i] for i in ri]
  return data


def train_epoch(epochNum):
    for num in range(epochNum):
        shuffleTrain = shufflelists(train)
        # 根据shuffle过的数据进行一次训练
        for k in range(len(train)):
            sess.run(train_step,feed_dict={inputs:shuffleTrain[k][0],labels:shuffleTrain[k][1]})
        testSampleLoss = 0
        trainSampleLoss = 0

        for i in range(len(test)):
            testSampleLoss += sess.run(loss,feed_dict={inputs:test[i][0],labels:test[i][1]})

        for j in range(len(train)):
            trainSampleLoss += sess.run(loss,feed_dict={inputs:train[j][0],labels:train[j][1]})

        # 展示每epoch一次后，在测试集和训练集上的损失值
        print(num,'train:',round(trainSampleLoss/83,3),'test:',round(testSampleLoss/20,3))




if __name__ == '__main__':

    rootPath = 'D:/workstation/repositories/nndeeplearning/data/super_smart/'
    rootPath = '/data/wangtd/workspace/'

    # 以numpy的方式加载数据，注意这里的编码方式
    mfc = np.load(rootPath + 'X.npy',encoding='bytes')
    art = np.load(rootPath + 'Y.npy',encoding='bytes')
    print('mfc shape:',mfc.shape)
    print('art shape:',art.shape)

    # 总的数据个数
    totalSamples = len(mfc)
    # print('total samples:',totalSamples)
    # 设置验证集个数
    validationSet = 0.2

    '''
        注意数据的输入形式:
        其中输入数据的形状是[n_samples, n_steps, D_input]
        其中输出数据的形状是[n_samples, D_output]
    '''

    ## 1 进行数据处理
    data = dataPreHandle(mfc,art)

    train = data[int(totalSamples*validationSet):]
    test = data[:int(totalSamples*validationSet)]

    ## 2 构建网络结构
    D_input = 39
    D_label = 24
    learning_rate = 7e-5
    num_units = 1024    # 隐藏层

    inputs = tf.placeholder(tf.float32,[None,None,D_input],name='inputs')
    labels = tf.placeholder(tf.float32,[None,D_label],name = 'labels')


    rnn_cell = LSTMcell(incoming=inputs,D_input=D_input,D_cell=num_units,initializer=orthogonal_initializer)

    rnn0 = rnn_cell.all_steps()
    # 将3维tensor [n_steps, n_samples, D_cell]转成 矩阵[n_steps*n_samples, D_cell]
    # 用于计算outputs
    rnn = tf.reshape(rnn0,[-1,num_units])

    # 定义输出层学习参数
    W = weight_init([num_units,D_label])    # [1024,24]
    b = bias_init([D_label])    # 24

    # 输出结果
    output = tf.matmul(rnn,W) + b
    # 损失（均方差）
    loss = tf.reduce_mean((output-labels)**2)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    ## 3 训练网络
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    startTime = time.time()
    # 训练数据
    train_epoch(10)
    endTime = time.time()
    print('%f seconds' % round(((endTime-startTime),2)))








