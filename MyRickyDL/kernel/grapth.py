class CalculateGrapth:
    def __init__(self):
        self.nodes = []
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)

    def clear_jacobi(self):
        """清除图中全部节点的雅克比矩阵"""
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        """重置图中全部节点的值"""
        for node in self.nodes:
            node.reset_value(False)

    def node_count(self):
        return len(self.nodes)

    def draw(self):
        #  try:
        #   import network
        a = 5


def_cal = CalculateGrapth()
