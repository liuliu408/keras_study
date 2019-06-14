import matplotlib.pyplot as plt
import numpy as np
a = [[1,2]]
a = np.array(a)

c = a[:,0]
b = a[:,1]

print (c ,b)

print (a.shape)

plt.scatter(c,b)
plt.show()