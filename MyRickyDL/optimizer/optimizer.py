import numpy as  np
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
        self.
        self.acc_no += 1
        