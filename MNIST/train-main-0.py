
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 权值初始化
def weight_variable(shape):
    initial = tf.random.truncated_normal(shape,stddev=0.1)
    # 生成一个截断的正态分布,其标准差为0.1
    return tf.Variable(initial)

# 偏置初始化 
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME") 
    # x为输入的tensor,其形状为[batch, in_height, in_width, in_channels],具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    # W为卷积核(滤波器),其形状为[filter_height, filter_width, in_channels, out_channels],具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    # strides[0]和strides[3]的两个1是默认值,strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding决定卷积方式,SAME会在外面补0

# 池化层(最大值池化)
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    # ksize[0]和ksize[3]默认为1,中间的2,2为池化窗口的大小
    # strides同conv2d,明显x,y方向的步长均为2

tf.compat.v1.disable_eager_execution()
# 定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32,[None,784])
y = tf.compat.v1.placeholder(tf.float32,[None,10])

# 改变x格式为4D的向量
x_image = tf.reshape(x,[-1,28,28,1]) 
# 第二个参数 : [batch, in_height, in_width, in_channels] [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
# 其中-1在程序运行后将会被赋值为100(即每批次中包含的图片的数量)

# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5,5,1,32]) 
# 5*5的采样窗口,32个卷积核(32个平面/32个通道)从一个平面(一个通道)提取特征
b_conv1 = bias_variable([32]) 
# 每一个卷积核一个偏置值

# 把x_image和权值向量进行卷积,再加上偏置值,并用relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
# x_image和W_conv1进行2D卷积操作再加上权值,最后输入relu激活函数

# 将第一卷积层输出进行池化
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5,5,32,64]) # 5*5的采样窗口,64个卷积核从32个平面提取特征
b_conv2 = bias_variable([64]) # 每一个卷积核一个偏置值

# 把h_pool1和权值向量进行卷积,再加上偏置值,并用relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

# 将第二卷积层输出进行池化
h_pool2 = max_pool_2x2(h_conv2)

# 说下这些卷积池化的过程(数据形状) : 
#   28*28的图片第一次卷积后还是28*28(SAME padding不会改变图片的大小),第一次池化后变为14*14
#   14*14的图片第二次卷积后还是14*14,第二次池化后变为7*7
#   经上述操作后最后获得64张7*7平面

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64,1024]) 
# 上一层共有7*7*64个像素点,全连接层1共1024个神经元
b_fc1 = bias_variable([1024])
# 每个神经元一个偏置值

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
# h_pool2为形状为[100(批次),7,7(高宽),64(图片或者通道数)]
# -1为任意值,计算时会处理为100
# 实际上就是将其后三个维度转化为7*7*64这一个维度

# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1) 

# 用keep_prob来表示神经元的输出概率(dropout)
keep_prob = tf.compat.v1.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 使用AdamOptimizer进行优化
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个bool型列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) # argmax返回一维张量中最大值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(11):
        for batch in range(n_batch):
            # 传入数据
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            # 用70%的神经元训练网络(dropout)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.})

        # 运行100%的神经元来检测网络准确率
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.})
        # 输出
        print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
       
    saver.save(sess, r"./itrain_cnn_net.ckpt")

