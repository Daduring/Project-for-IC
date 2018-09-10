import numpy as np
data=np.random.randint(10,size=10)
print(data)

print(data[-1])
data2=np.append(data[-1],data)
print(data2)
print(data2.shape)

data=np.array([x+1 for x in data])
print(data)
print(data.shape)
import tensorflow as tf
a=tf.random_uniform(
        (6,6), minval=-0.5,
        maxval=0.5, dtype=tf.float32)
print(type(a))

