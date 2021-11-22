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
    pass


class Add(Operator):
    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))

        for parent in self.parents:
            self.value += parent.value

    def get_jacobi(self, parent):

        return np.mat(np.eye(self.dim()))

