import numpy as np

a=np.array([1,2,3]).reshape(1,3)
b=np.array([2,3,4]).reshape(3,1)
c=b*a
print(c)