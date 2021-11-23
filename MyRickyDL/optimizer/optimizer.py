import abc

import numpy as np
from ..kernel import Node, Variable, get_node_from_graph
from ..kernel.grapth import CalculateGrapth


class Optimizer(object):

    def __init__(self, graph, target, learning_rate=0.1):

        assert isinstance(target, Node) and isinstance(graph, CalculateGrapth)
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):

        """add Gradient of samples"""
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        """average gradient"""
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):

        for node, gradient in node_gradients_dict.item():
            if isinstance(node, Node):
                pass

            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def forward_backward(self):

        self.graph.clear_jacobi()
        self.target.forward()

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                gradient = node.jacobiMatrix.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient

    @abc.abstractmethod
    def _update(self):
        """abstract function"""

    def updata(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)


class Adam(Optimizer):

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        assert 0.0 < beta_1 < 1.0

        # 历史梯度衰减系数
        self.beta_1 = beta_1
        assert 0.0 < beta_2 < 1.0

        self.beta_2 = beta_2

        # 历史梯度累计
        self.v = dict()

        # 历史梯度各个分量平方累积
        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                gradient = self.get_gradient(node)
                if node not in self.s:
                    self.v[node] = gradient
                    # np.power 求x的y次方
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.v[node] = self.beta_1 * self.v[node] + (1 - self.beta_1) * gradient

                    # 各个分量平方累积
                    self.s[node] = self.beta_2 * self.s[node] + (1 - self.beta_2) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate * self.v[node] / np.sqrt(self.s[node] + 1e-10))

                