import numpy as np

# a=np.array([1,2,3]).reshape(1,3)
# b=np.array([2,3,4]).reshape(3,1)
# c=b*a
# print(c)

# 用计算图搭建ADALINE并进行训练
import rickydl as ri

"""
制造并训练样本，根据均值为171，标准差为6 的正态分布采样500个男性身高， 并根据均值158，标准差为5的 正态分布采样500个女性身高

"""
male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [0] * 500

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T

# print(train_set.shape)
np.random.shuffle(train_set)

x = ri.kernel.Variable(dim=(3,1), init=False, trainable=False)

labels = ri.kernel.Variable(dim=(1,1), init=False, trainable=False)

w = ri.kernel.Variable(dim=(1,3), init=True, trainable=True)
b = ri.kernel.Variable(dim=(1,3), init=True, trainable=True)

output = ri.operate.Add(ri.operate.MatMul(w, x), b)
predict = ri.operate.Step()