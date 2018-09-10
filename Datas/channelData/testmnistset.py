import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size=10
x,y=mnist.train.next_batch(batch_size)
print(type(x))
print(x.shape)
for i,sample in enumerate(x):
	plt.imshow(sample.reshape(28,28),cmap='Greys_r')
	plt.show()
# plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')