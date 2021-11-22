from .nodes import Variable
from .grapth import def_cal


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = def_cal
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


def get_trainable_variables(node_name=None, name_scope=None, graph=None):
    """从graph中获取variable"""
    if graph is None:
        graph = def_cal
    if node_name is None:
        return [node for node in def_cal.nodes if isinstance(node, Variable) and node.trainable]


def update_node_value(node_name, new_value, name_scope=None, graph=None):
    node = get_node_from_graph(node_name, name_scope, graph)
    assert node is not None

    assert node.value.shape == new_value.shape
    node.value = new_value.shape


class Name_Scope(object):
    def __init__(self, name_scope):
        self.name_scope = name_scope

    def __enter__(self):
        def_cal.name_scope = self.name_scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        def_cal.name_scope = None

