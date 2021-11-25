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
predict = ri.operate.Step(output)
# 损失函数
loss = ri.operate.loss.PerceptionLoss(ri.operate.MatMul(labels, output))
learning_rate = 0.0001

for epoch in range(50):

    for i in range(len(train_set)):

        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1]).T

        x.set_value(features)
        labels.set_value(l)

        loss.forward()

        w.backward(loss)

        b.backward(loss)
        w.set_value(w.value - learning_rate * w.jacobiMatrix.T.reshape(w.shape()))
        b.set_value(b.value - learning_rate * b.jacobiMatrix.T.reshape(b.shape()))
        ri.default_cal.clear_jacobi()

    pred = []

    for i in range(len(train_set)):

        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)

        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value[0, 0])  # 模型的预测结果：1男，0女

    pred = np.array(pred) * 2 - 1  # 将1/0结果转化成1/-1结果，好与训练标签的约定一致

    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)

    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))

