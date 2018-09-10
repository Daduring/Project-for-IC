import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
import math
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
	def __init__(self, generator1,generator2, discriminator, data):
		self.generator1 = generator1
		self.generator2 = generator2
		self.discriminator = discriminator
		self.data = data
		self.batch_size = 128

		# data
		self.z_dim = self.data.z_dim 
		self.y_dim = self.data.y_dim # condition 10
		self.X_dim = self.data.X_dim 

		self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.X_dim])#real channel output
		self.z1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])#random numbers to generate h
		self.z2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])#random numbers to generate w
		self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.y_dim])#channel input repeated y_dim times
		self.y_stack=tf.placeholder(tf.float32,shape=[self.batch_size,self.X_dim])#channel input repeated z_dim times
		self.weight=tf.placeholder(tf.float32,shape=[])
		# nets
		self.G1_sample = self.generator1(self.z1)#generated h
		self.G2_sample = self.generator2(self.z2)#generated w
		self.G_sample=tf.multiply(self.G1_sample,self.y_stack)+self.G2_sample#h*x+w
		#self.G_sample_sorted,_=tf.nn.top_k(self.G_sample,self.X_dim)
		self.X_sorted,_=tf.nn.top_k(self.X,self.X_dim)

		self.D_real, _ = self.discriminator(concat(self.X_sorted, self.y))
		self.D_fake, _ = self.discriminator(concat(self.G_sample, self.y), reuse = True)
		
		# loss
		self.D_loss_real = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.D_real, tf.ones_like(self.D_real)), 1))
		self.D_loss_fake=self.weight*tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.D_fake, tf.zeros_like(self.D_fake)), 1)) 
		self.D_loss=self.D_loss_real+self.D_loss_fake
		self.G_loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.D_fake, tf.ones_like(self.D_fake)), 1))

		# solver
		self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator2.vars+self.generator1.vars)#G1,G2的vars
	
		for var in self.discriminator.vars:
			print (var.name)
		for var in self.generator1.vars:
			print (var.name)
			
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 128):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())

		weight=1
		for epoch in range(training_epoches):
			# update D
			X_b,y_b,y_stack_b = self.data(batch_size)
			D_loss_real=self.sess.run(
				self.D_loss_real,
				feed_dict={self.X: X_b, self.y: y_b, self.y_stack: y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)}
				)
			D_loss_fake=self.sess.run(
				self.D_loss_fake,
				feed_dict={self.weight:weight,self.X: X_b, self.y: y_b, self.y_stack: y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)}
				)
			#weight=D_loss_real+1/math.log(epoch+10)
			self.sess.run(
				self.D_solver,
				feed_dict={self.weight:weight,self.X: X_b, self.y: y_b, self.y_stack: y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)}
				)
			# update G
			k = 1
			for _ in range(k):
				self.sess.run(
					self.G_solver,
					feed_dict={self.y: y_b,self.y_stack:y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)}
				)
			
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr = self.sess.run(
						self.D_loss,
            			feed_dict={self.weight:weight,self.X: X_b, self.y: y_b, self.y_stack: y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)})
				G_loss_curr = self.sess.run(
						self.G_loss,
						feed_dict={self.y: y_b,self.y_stack:y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

			if epoch % 1000 == 0:
				# y_s = sample_y(16, self.y_dim, fig_count%10)
				# samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

				# fig = self.data.data2fig(samples)
				# plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
				# fig_count += 1
				# plt.close(fig)
				# generated_samples=self.sess.run(
				# 	self.G_sample,
				# 	feed_dict={self.y_stack:y_stack_b, self.z1: sample_z(batch_size, self.z_dim),self.z2: sample_z(batch_size, self.z_dim)})
				#real_samples=X_b[0,:]
				#画频率分布直方图
				# fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
				#第二个参数是柱子宽一些还是窄一些，越大越窄越密  
				# print(generated_samples[0,:].shape)
				# print(real_samples.shape)
				# for var in self.discriminator.vars:
				# 	print (var.name)
				# for var in self.generator1.vars:
				# 	print(var.name)
				# for var in self.generator2.vars:
				# 	print (var.name)

				# print(generated_samples[0,:])
				#print(y_stack_b[0,0])

				generated_data_samples=self.sess.run(
					self.G_sample,
					feed_dict={self.y_stack:y_stack_b,self.z1: sample_z(self.batch_size, self.z_dim),self.z2: sample_z(self.batch_size, self.z_dim)})


				generated_h_samples=self.sess.run(
					self.G1_sample,
					feed_dict={self.z1: sample_z(self.batch_size, self.z_dim)})
				generated_w_samples=self.sess.run(
					self.G2_sample,
					feed_dict={self.z2: sample_z(self.batch_size, self.z_dim)})
				generated_h_m=generated_h_samples[0:20,:]
				generated_h=generated_h_m.reshape([1,15680])
				generated_w_m=generated_w_samples[0:20,:]
				generated_w=generated_w_m.reshape([1,15680])

				fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
				#第二个参数是柱子宽一些还是窄一些，越大越窄越密  

				#print(generated_h_samples[0,:])
				ax0.hist(generated_h[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
				##pdf概率分布图，一千个数落在某个区间内的数有多少个  
				ax0.set_title('h')
				ax1.hist(generated_w[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
				##pdf概率分布图，一千个数落在某个区间内的数有多少个  
				ax1.set_title('w')
				fig.subplots_adjust(hspace=0.4)  
				plt.show()


				if epoch % 1000 == 0:
					self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan.ckpt"))
					np.savetxt("generated_h.txt", generated_h_samples)
					np.savetxt("generated_w.txt", generated_w_samples)
					np.savetxt("generated_data.txt", generated_data_samples)


if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# save generated images
	sample_dir = 'Samples/mnist_cgan_mlp'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator1 = G_mlp_mnist()
	generator2=G_mlp_hxwn()
	discriminator = D_mlp_mnist()

	#data = channel_hdx_plus_wn('mlp')
	data=channel_h10x_plus_wn('mlp')

	# run
	cgan = CGAN(generator1,generator2, discriminator, data)
	# sess=cgan.sess
	# cgan.saver.restore(sess,os.path.join('ckpt','cgan.ckpt'))
	cgan.train(sample_dir)
	# w=sess.graph.get_tensor_by_name('G_mlp_mnist/fully_connected/weights:0')
	# w_value=sess.run(w)
	# #print(w_value)
	# np.savetxt(os.path.join('1hx1wnckpt','w_value.txt'), w_value)

	# b=sess.graph.get_tensor_by_name('G_mlp_mnist/fully_connected/biases:0')
	# b_value=sess.run(b)
	# #print(w_value)
	# np.savetxt(os.path.join('1hx1wnckpt','b_value.txt'), b_value)

	# k=sess.graph.get_tensor_by_name('G_mlp_mnist/fully_connected_1/weights:0')
	# k_value=sess.run(k)
	# #print(w_value)
	# np.savetxt(os.path.join('1hx1wnckpt','k_value.txt'), k_value)

	# t=sess.graph.get_tensor_by_name('G_mlp_mnist/fully_connected_1/biases:0')
	# t_value=sess.run(t)
	# #print(w_value)
	# np.savetxt(os.path.join('1hx1wnckpt','t_value.txt'), t_value)


	# generated_h_samples=sess.run(
	# 	cgan.G1_sample,
	# 	feed_dict={cgan.z1: sample_z(cgan.batch_size, cgan.z_dim)})
	# generated_w_samples=sess.run(
	# 	cgan.G2_sample,
	# 	feed_dict={cgan.z2: sample_z(cgan.batch_size, cgan.z_dim)})
	# generated_h_m=generated_h_samples[0:20,:]
	# generated_h=generated_h_m.reshape([1,15680])
	# generated_w_m=generated_w_samples[0:20,:]
	# generated_w=generated_w_m.reshape([1,15680])

	# fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
	# #第二个参数是柱子宽一些还是窄一些，越大越窄越密  

	# #print(generated_h_samples[0,:])
	# ax0.hist(generated_h[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
	# ##pdf概率分布图，一千个数落在某个区间内的数有多少个  
	# ax0.set_title('h')
	# ax1.hist(generated_w[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
	# ##pdf概率分布图，一千个数落在某个区间内的数有多少个  
	# ax1.set_title('w')
	# fig.subplots_adjust(hspace=0.4)  
	# plt.show()
