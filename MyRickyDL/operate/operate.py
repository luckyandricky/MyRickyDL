from abc import ABC

from ..kernel import Node
import numpy as np


def fill_diagonal(to_be_filled, filler):
    assert to_be_filled.shape[0] / \
           filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]

    n = int(to_be_filled.shape[0]) / filler.shape[0]

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node, ABC):

    """抽象类"""
    """其实什么也没做，只是起到整理类继承结构作用"""
    pass


class Add(Operator):
    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):

        return np.mat(np.eye(self.dim()))


class MatMul(Operator):
    def compute(self):
        """判断两个矩阵相乘是否符合要求"""
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        zeros = np.mat(np.zeros((self.dim(),parent.dim())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dim()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dim()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Softmax(Operator):

    @staticmethod
    def softmax(a):
        # 防止指数过大
        a[a > 1e2] = 1e2
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = Softmax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        raise NotImplementedError("Don`t use softmax`s get_jacobi")

