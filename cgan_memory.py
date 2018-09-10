import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import os,sys

sys.path.append('utils')
from nets import *
from datas import *

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

# for test
def sample_y(m, n, ind):
	y = np.zeros([m,n])
	for i in range(m):
		y[i,ind] = 1
	return y

def concat(z,y):#z和y联合
	return tf.concat([z,y],1)

class CGAN():
	def __init__(self, generator, discriminator, data):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data
		self.batch_size = 64

		# data
		self.z_dim = self.data.z_dim 
		self.X_dim = self.data.X_dim 
		self.memory_dim=self.data.memory_dim #number of former input to be used 5

		self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#real channel output
		self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])#random numbers to generate w
		self.y0 = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#channel input at t-0 shape[batch_size,X_dim],same digits one row
		self.y1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#channel input at t-1
		self.y2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#channel input at t-2
		self.y3 = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#channel input at t-3
		self.y4 = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#channel input at t-4
		
		self.y_stack=tf.placeholder(tf.float32,shape=[self.batch_size,self.memory_dim])#channel input condition to feed to G

		# nets
		self.a0=tf.Variable(tf.random_normal([1], mean=0.5,stddev=0.02),name='a0')
		self.a1=tf.Variable(tf.random_normal([1,], mean=0.5,stddev=0.02),name='a1')
		self.a2=tf.Variable(tf.random_normal([1,], mean=0.5,stddev=0.02),name='a2')
		self.a3=tf.Variable(tf.random_normal([1,], mean=0.5,stddev=0.02),name='a3')
		self.a4=tf.Variable(tf.random_normal([1,], mean=0.5,stddev=0.02),name='a4')

		self.G_sample_w = self.generator(self.z)#generated w

		self.a0y0=tf.multiply(self.y0,self.a0)
		self.a1y1=tf.multiply(self.y1,self.a1)
		self.a2y2=tf.multiply(self.y2,self.a2)
		self.a3y3=tf.multiply(self.y3,self.a3)
		self.a4y4=tf.multiply(self.y4,self.a4)

		self.X_sample=tf.add_n([self.a0y0,self.a1y1,self.a2y2,self.a3y3,self.a4y4,self.G_sample_w])#h*x+w
		#self.G_sample_sorted,_=tf.nn.top_k(self.G_sample,self.X_dim)
		self.X_sorted,_=tf.nn.top_k(self.X,self.X_dim)

		self.D_real, _ = self.discriminator(concat(self.X_sorted, self.y_stack))
		self.D_fake, _ = self.discriminator(concat(self.X_sample, self.y_stack), reuse = True)
		
		# loss
		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

		# solver
		self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
		print(self.generator.vars)
		print(self.a0)
		self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator.vars+[self.a0,self.a1,self.a2,self.a3,self.a4])
	
		for var in self.discriminator.vars:
			print (var.name)
			
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckptMemory', training_epoches = 1000000, batch_size = 64):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		
		for epoch in range(training_epoches):
			# update D
			X_b,y0_b,y1_b,y2_b,y3_b,y4_b,y_stack_b = self.data(batch_size)
			self.sess.run(
				self.D_solver,
				feed_dict={self.X: X_b, self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)}
				)
			# update G
			k = 1
			for _ in range(k):
				self.sess.run(
					self.G_solver,
					feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)}
				)
			
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr = self.sess.run(
						self.D_loss,
            			feed_dict={self.X: X_b, self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				G_loss_curr = self.sess.run(
						self.G_loss,
						feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

			if epoch % 10000 == 0:
				# y_s = sample_y(16, self.y_dim, fig_count%10)
				# samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

				# fig = self.data.data2fig(samples)
				# plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
				# fig_count += 1
				# plt.close(fig)
				generated_samples=self.sess.run(
					self.X_sample,
					feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				real_samples=X_b[0,:]
				#画频率分布直方图
				a0=self.sess.run(
					self.a0,
					feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				a1=self.sess.run(
					self.a1,
					feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				a2=self.sess.run(
					self.a2,
					feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				a3=self.sess.run(
					self.a3,
					feed_dict={self.y0: y0_b, self.y1: y1_b,self.y2: y2_b,self.y3: y3_b,self.y4: y4_b,self.y_stack: y_stack_b, self.z: sample_z(batch_size, self.z_dim)})
				print(a0)
				print(a1)
				print(a2)
				print(a3)
				fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
				#第二个参数是柱子宽一些还是窄一些，越大越窄越密  
				# print(generated_samples[0,:].shape)
				# print(real_samples.shape)
				# for var in self.discriminator.vars:
				# 	print (var.name)
				# for var in self.generator1.vars:
				# 	print(var.name)
				# for var in self.generator2.vars:
				# 	print (var.name)

				#print(generated_samples[0,:])
				#print(y_stack_b[0,:])
				ax0.hist(generated_samples[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
				##pdf概率分布图，一千个数落在某个区间内的数有多少个  
				ax0.set_title('pdf_fake')
				ax1.hist(real_samples,50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
				##pdf概率分布图，一千个数落在某个区间内的数有多少个  
				ax1.set_title('pdf_real')
				fig.subplots_adjust(hspace=0.4)  
				plt.show()


				if epoch % 2000 == 0:
					self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan.ckpt"))


if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# save generated images
	sample_dir = 'Samples/mnist_cgan_mlp'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_mlp_mnist()
	discriminator = D_mlp_mnist()

	#data = channel_hdx_plus_wn('mlp')
	data=channel_with_memory('mlp')

	# run
	cgan = CGAN(generator, discriminator, data)
	cgan.train(sample_dir)
	# sess=cgan.sess
	# cgan.saver.restore(sess,os.path.join('1hx1wnckpt','cgan.ckpt'))
	# generated_h_samples=sess.run(
	# 	cgan.G1_sample,
	# 	feed_dict={cgan.z1: sample_z(cgan.batch_size, cgan.z_dim)})
	# generated_w_samples=sess.run(
	# 	cgan.G2_sample,
	# 	feed_dict={cgan.z2: sample_z(cgan.batch_size, cgan.z_dim)})

	# fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
	# #第二个参数是柱子宽一些还是窄一些，越大越窄越密  

	# print(generated_h_samples[0,:])
	# ax0.hist(generated_h_samples[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
	# ##pdf概率分布图，一千个数落在某个区间内的数有多少个  
	# ax0.set_title('h')
	# ax1.hist(generated_w_samples[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
	# ##pdf概率分布图，一千个数落在某个区间内的数有多少个  
	# ax1.set_title('w')
	# fig.subplots_adjust(hspace=0.4)  
	# plt.show()
