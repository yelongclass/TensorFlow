# encoding:utf-8
# Python2 兼容
from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np
##《TF Girls 修炼指南》第四期
# 正式开始机器学习
# 首先我们要确定一个目标: 图像识别

# 我这里就用Udacity Deep Learning的作业作为辅助了

# 1. 下载数据  http://ufldl.stanford.edu/housenumbers/


def reformat(samples,labels):
	# 改变原始数据的形状
	#
	#（图片高，图片宽，通道数，图片数）--> （图片数，图片高，图片宽，通道数）
	# labels 变成 one-hot ecoding, [2]=[0,0,1,0,0,0,0,0,0,0,0]
	# digit 0 represented as 10
	# labels 变成 one-hot ecoding, [10]=[1,0,0,0,0,0,0,0,0,0,0]
	samples = np.transpose(samples,(3,0,1,2)).astype(np.float32)
	labels = np.array([x[0] for x in labels])
	one_hot_labels = []
	for num in labels:
		one_hot = [0.0]*10
		if num == 10:
			one_hot[0] = 1.0
		else:
			one_hot[num] = 1.0
		one_hot_labels.append(one_hot)
	labels = np.array(one_hot_labels).astype(np.float32)
	return samples, labels

def normalize(samples):
	'''
	并且灰度化: 从三色通道 -> 单色通道     省内存 + 加快训练速度
	(R + G + B) / 3
	将图片从 0 ~ 255 线性映射到 -1.0 ~ +1.0
	@samples: numpy array
	'''
	#Grayscaleing images
	a = np.add.reduce(samples, keepdims=True, axis=3)  # shape (图片数，图片高，图片宽，通道数)
	a = a/3.0
	#normalized
	return a/128.0 - 1.0


def inspect(dataset, labels, i):
	# 显示图片看看
	if dataset.shape[3] == 1:
		shape = dataset.shape
		dataset = dataset.reshape(shape[0], shape[1], shape[2])
	print(labels[i])
	plt.imshow(dataset[i])
	plt.show()

def distribution(labels, name):
	# 查看一下每个label的分布，再画个统计图
	# keys:
	# 0
	# 1
	# 2
	# ...
	# 9
	count = {}
	for label in labels:
		#if key in count 
		key = 0 if label[0] == 10 else label[0]
		if key in count:
			count[key] += 1
		else:
			count[key] = 1
	x = []
	y = []
	for k, v in count.items():
		# print(k, v)
		x.append(k)
		y.append(v)

	y_pos = np.arange(len(x))
	plt.bar(y_pos, y, align='center', alpha=0.5)
	plt.xticks(y_pos, x)
	plt.ylabel('Count')
	plt.title(name + ' Label Distribution')
	plt.show()


#load data
traindata = load('data/train_32x32.mat')
testdata = load('data/test_32x32.mat')
#extradata = load('data/extra_32x32.mat')

print('Train Data Samples Shape:', traindata['X'].shape)
print('Train Data Labels Shape:', traindata['y'].shape)

print('Test Data Samples Shape:', testdata['X'].shape)
print('Test Data Labels Shape:', testdata['y'].shape)

#print('Extra Data Samples Shape:', extradata['X'].shape)
#print('Extra Data Labels Shape:', extradata['y'].shape

# 3. Pre-Process Data
#Get samples and Lables
train_samples = traindata['X']
train_labels = traindata['y']
test_samples = testdata['X']
test_labels = testdata['y']
# test_samples = extradata['X']
# test_labels = extradata['y']

_train_samples, _train_labels = reformat(train_samples, train_labels)
_test_samples, _test_labels = reformat(test_samples, test_labels)

#global configuration / hyper parameters
num_labels = 10
image_size = 32
num_channels = 1

if __name__ == '__main__':
# 2. 探索数据
	#inspect(_train_samples, _train_labels, 123)
	#_train_samples = normalize(_train_samples)
	#inspect(_train_samples, _train_labels, 123)
	distribution(train_labels, 'Train Labels')
	distribution(test_labels, 'Test Labels')
# 4. 构建一个基本网络, 基本的概念+代码 ， TensorFlow的世界
# 5. 卷积ji
# 6. 来实验吧
# 7. 微调与结果
