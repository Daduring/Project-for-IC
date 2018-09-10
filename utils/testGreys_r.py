import numpy as np
import matplotlib.pyplot as plt
sample=np.random.random((28,28))+5
plt.imshow(sample, cmap='Greys_r')
plt.show()