import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import time

tf.compat.v1.disable_eager_execution()


# 定义一个确认时间的函数:年-月-日 时:分:秒
def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))


CONTENT_IMG = 'content.jpg'
STYLE_IMG = 'style5.jpg'
OUTPUT_DIR = 'neural_style_transfer_tensorflow\\'

# 如果没有OUTPUT_DIR路径对应的文件夹则新建一个
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# 图片格式设置：宽800，高300，颜色通道3
IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

# 噪声比例
NOISE_RATIO = 0.7

# LOSS = LossStyle + LossContent
# 内容损失与风格损失的权重
BETA = 5
ALPHA = 100

# VGG-19神经网络模型
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

# 平均值，用以求后续的平均方差等，并改成1*1*1*3数组
MEAN_VALUES = np.array([123.68, 116.799, 103.939]).reshape((1, 1, 1, 3))


# 加载VGG-19的函数:通过给点路径加载已经训练好的vgg模型
def load_vgg_model(path):
    """
    Details of the VGG19 model:
    - 0 is conv1_1 (3, 3, 3, 64)           # 2D卷积：3*3的卷积核，深度3（输入通道数），步长为3，64个内核（输出64通道数）
	- 1 is relu
	- 2 is conv1_2 (3, 3, 64, 64)
	- 3 is relu
	- 4 is maxpool
	- 5 is conv2_1 (3, 3, 64, 128)
	- 6 is relu
	- 7 is conv2_2 (3, 3, 128, 128)
	- 8 is relu
	- 9 is maxpool                          # 最大池化层
	- 10 is conv3_1 (3, 3, 128, 256)
	- 11 is relu
	- 12 is conv3_2 (3, 3, 256, 256)
	- 13 is relu
	- 14 is conv3_3 (3, 3, 256, 256)
	- 15 is relu
	- 16 is conv3_4 (3, 3, 256, 256)
	- 17 is relu
	- 18 is maxpool
	- 19 is conv4_1 (3, 3, 256, 512)
	- 20 is relu
	- 21 is conv4_2 (3, 3, 512, 512)
	- 22 is relu
	- 23 is conv4_3 (3, 3, 512, 512)
	- 24 is relu
	- 25 is conv4_4 (3, 3, 512, 512)
	- 26 is relu
	- 27 is maxpool
	- 28 is conv5_1 (3, 3, 512, 512)
	- 29 is relu
	- 30 is conv5_2 (3, 3, 512, 512)
	- 31 is relu
	- 32 is conv5_3 (3, 3, 512, 512)
	- 33 is relu
	- 34 is conv5_4 (3, 3, 512, 512)
	- 35 is relu
	- 36 is maxpool
	- 37 is fullyconnected (7, 7, 512, 4096)    # 全连接层
	- 38 is relu
	- 39 is fullyconnected (1, 1, 4096, 4096)
	- 40 is relu
	- 41 is fullyconnected (1, 1, 4096, 1000)
	- 42 is softmax
    """

    vgg = np.loadtxt(path)
    vgg_layers = vgg['layers']

    # 提取模型的权重 W与偏置 b
    def _weights(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    # 二维卷积 + relu函数
    def _conv2d_relu(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)  # 获取训练好的权重，不希望被训练(constant)
        b = tf.constant(np.reshape(b, (b.size)))  # 获取训练好的偏置，不希望被训练(constant)
        return tf.nn.relu(tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

    # 平均池化层
    def _avg_pool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 定义一个空字典graph，用来放置VGG模型训练顺序
    graph = {}
    # 将各个步骤放入字典中，最终得到vgg模型的全部过程
    graph['input'] = tf.variable(np.zeros((1, IMAGE_H, IMAGE_W, COLOR_C)), dtype='float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avg_pool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avg_pool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avg_pool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avg_pool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avg_pool(graph['conv5_4'])
    return graph


# 内容损失函数
def content_loss_func(sess, model):
    def _content_loss(p, x):
        N = p.shape[3]  # 图片的深度
        M = p.shape[1] * p.shape[2]  # 图片的高度×宽度
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))  # 内容损失函数公式定义

    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])  # 将卷积到conv4_2时的结果计算内容损失值


# 图片经过每层处理后，由写实(conv1_1)到抽象(conv5_1)的权重分布
STYLE_LAYERS = [('conv1_1', 0.5), ('conv2_1', 1.0), ('conv3_1', 1.5), ('conv4_1', 3.0), ('conv5_1', 4.0)]


# 风格损失函数定义
def style_loss_func(sess, model):
    def _gram_matrix(F, N, M):  # 风格损失梯度矩阵
        Ft = tf.reshape(F, (M, N))  # 以(M, N)为基础变化矩阵
        return tf.matmul(tf.transpose(Ft), Ft)  # 转置，与自身求内积

    def _style_loss(a, x):  # 风格损失函数计算公式：a为风格图，x为生成图
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]
        A = _gram_matrix(a, N, M)
        G = _gram_matrix(x, N, M)
        return (1 / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow(G - A, 2))

    # 将图片经过每一层（共五层）的风格损失按照权重分布相加，得到最终得风格损失值
    return sum([_style_loss(sess.run(model[layer_name]), model[layer_name]) * w for layer_name, w in STYLE_LAYERS])


# 由内容图，随机产生一张初始图片
def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, COLOR_C)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


# 加载图片
def load_image(path):
    image = scipy.misc.imread(path)  # 加载图片（3维（宽高颜色））
    image = scipy.misc.imresize(image, (IMAGE_H, IMAGE_W))  # 按照标准高宽进行变换
    image = np.reshape(image, ((1,) + image.shape))  # 将3维tensor变成4维tensor
    image = image - MEAN_VALUES
    return image


# 保存图片
def save_image(path, image):
    image = image + MEAN_VALUES  # 将处理的图片加回原先MEAN_VALUES
    image = image[0]  # 此时image为4维tensor，获取索引0可以取出3维tensor（原图片规格）
    image = np.clip(image, 0, 255).astype('uint8')  # 将内容限制在0~255之间并改为正数
    scipy.misc.imsave(path, image)  # 按路径保存图片


# 开始训练
the_current_time()  # 记录开始训练时刻

with tf.compat.v1.Session() as sess:  # 将tf.Session()定义为sess，使得后续可以进行sess.run()
    content_image = load_image(CONTENT_IMG)
    style_image = load_image(STYLE_IMG)
    model = load_vgg_model(VGG_MODEL)

    input_image = generate_noise_image(content_image)
    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(content_image))
    content_loss = content_loss_func(sess, model)

    sess.run(model['input'].assign(style_image))
    style_loss = style_loss_func(sess, model)

    total_loss = BETA * content_loss + ALPHA * style_loss
    optimizer = tf.train.AdamOptimizer(2.0)
    train = optimizer.minimize(total_loss)

    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))

    ITERATIONS = 2000
    for i in range(ITERATIONS):
        sess.run(train)
        if i % 100 == 0:
            output_image = sess.run(model['input'])
            the_current_time()
            print('Iteration %d' % i)
            print('Cost: ', sess.run(total_loss))

            save_image(os.path.join(OUTPUT_DIR, 'output_%d.jpg' % i), output_image)
