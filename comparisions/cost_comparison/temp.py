import matplotlib.pyplot as plt
import numpy as np


x = range(1, 10000000, 100)
y1 = [np.sqrt(i) for i in x]
y2 = [np.log10(i) for i in x]

plt.loglog(x, x, label='linear')
plt.loglog(x, y1, label='sqrt')
plt.loglog(x, y2, label='log')
plt.legend()
plt.show()
