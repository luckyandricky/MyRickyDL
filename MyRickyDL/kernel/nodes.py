import numpy as np


class Nodes(object):
    def __init__(self, *parents, **kargs):
        # 加了星号（ ** ）的变量名会存放所有未命名的变量参数
        # 加了星号（ * ）的变量名会存放所有未命名的变量参数，不能存放dict，否则报错。
        self.kargs = kargs
        self.gtapth = kargs.get('grapth', default_grapth)
        self.gen
