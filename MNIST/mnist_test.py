import tensorflow as tf 
from cv2 import cv2
import numpy as np


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

# 池化层(最大值池化)
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# 网络模型
def network(x):

    # 改变x格式为4D的向量
    x_image = tf.reshape(x,[-1,28,28,1]) 

    # 把x_image和权值向量进行卷积,再加上偏置值,并用relu激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

    # 将第一卷积层输出进行池化
    h_pool1 = max_pool_2x2(h_conv1)

    # 把h_pool1和权值向量进行卷积,再加上偏置值,并用relu激活函数
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

    # 将第二卷积层输出进行池化
    h_pool2 = max_pool_2x2(h_conv2)

    # 把池化层2的输出扁平化为1维
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

    # 求第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1) 

    # 使用drop
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    # 计算输出
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    return prediction

tf.compat.v1.disable_eager_execution()
# 定义两个placeholder
x = tf.compat.v1.placeholder(tf.float32,[None,784])
y = tf.compat.v1.placeholder(tf.float32,[None,10])

# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5,5,1,32]) 
b_conv1 = bias_variable([32]) 

# 初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5,5,32,64]) 
b_conv2 = bias_variable([64]) 

# 初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64,1024]) 
b_fc1 = bias_variable([1024])

# 用keep_prob来表示神经元的输出概率(dropout)
keep_prob = tf.compat.v1.placeholder(tf.float32)

# 初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

# 开启会话并导入模型
sess = tf.compat.v1.InteractiveSession()
saver = tf.compat.v1.train.Saver()
saver.restore(sess, r"./itrain_cnn_net.ckpt")

# 开启摄像头
cap = cv2.VideoCapture(0)

while(1):

    # 读取摄像头
    ret, frame = cap.read()

    # 在图像上绘制矩形框(红色)
    cv2.rectangle(frame,(320,150),(390,220),(0,0,255),2)
    cv2.imshow("capture", frame)

    # 确定ROI
    roiImg = frame[150:320,220:390]

    # 将该区域图像改变为CNN输入格式
    img = cv2.resize(roiImg,(28,28))

    # 将3通道改为单通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 改为float32格式
    np_img = img.astype(np.float32)

    # 连接网络
    netoutput = network(np_img)
    
    # 获取网络输出
    prediction = sess.run(netoutput,feed_dict={keep_prob:0.})
    
    # 转换为列表并取结果
    predicts = prediction.tolist() 
    label = predicts[0]
    result = label.index(max(label))
    print('result num:')
    print(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

