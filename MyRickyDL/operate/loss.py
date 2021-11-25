import numpy as np
from ..kernel import Node
from .operate import Softmax


class LossFun(Node):
    pass


class LogLoss(LossFun):
    def compute(self):
        assert len(self.parents) == 1

        x = self.parents[0].value

        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))

    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFun):
    def compute(self):
        prob = Softmax.softmax(self.parents[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))

    def get_jacobi(self, parent):
        # 这里存在重复计算，但为了代码清晰简洁，舍弃进一步优化
        prob = Softmax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
