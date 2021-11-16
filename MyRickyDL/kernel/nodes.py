import numpy as np
import abc
from grapth import CalculateGrapth, def_cal


class Node(object):
    def __init__(self, *parents, **kargs):
        # 加了星号（ ** ）的变量名会存放所有未命名的变量参数
        # 加了星号（ * ）的变量名会存放所有未命名的变量参数，不能存放dict，否则报错。
        self.kargs = kargs
        self.grapth = kargs.get('grapth', def_cal)
        self.gen_node_name(**kargs)
        self.parents = list(parents)
        self.child = []
        self.value = None
        self.jacobiMatrix = None  # jacobi矩阵
        for parents in self.parents:
            parents.child.append(self)

        self.grapth.add_node(self)

    @abc.abstractmethod
    def compute(self):
        """abstract function"""

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """abstract function"""

    def dim(self):
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        return self.value.shape

    def reset_value(self, recursive=True):
        self.value = None
        if recursive:
            for child in self.child:
                child.reset_value()

    def get_parents(self):
        return self.parents

    def get_chile(self):
        return self.child

    def gen_node_name(self, **kargs):
        self.name = kargs.get('name', '{}:{}'.format(self.__class__.__name__, self.grapth.node_count()))
        if self.grapth.name_scope:
            self.name = '{}/{}'.format(self.grapth.name_scope, self.name)

    def forward(self):
        for node in self.parents:
            if node.value is None:
                node.forward()

        self.compute()

    def backward(self, res):
        if self.jacobiMatrix is None:
            if self is res:
                self.jacobiMatrix = np.mat(np.eye(self.dim()))

            else:
                self.jacobiMatrix = np.mat(np.zeros((res.dim(), self.dim())))

                for child in self.get_chile():
                    if child.value is not None:
                        self.jacobiMatrix += child.backward(res) * child.get_jacobi(self)

        return self.jacobiMatrix

    def clear_jacobi(self):
        self.jacobiMatrix = None





















