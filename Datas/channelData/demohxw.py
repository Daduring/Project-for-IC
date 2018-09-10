import tensorflow as tf
import tensorflow.contrib.layers  as tcl
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import random

# def concat(z,y):
# 	return tf.concat([z,y],1)#横着连接，即第一维大小不变

g_random_dim=100
condition_dim=10
g_input_dim=g_random_dim#+condition_dim#generator的输入randomn number的个数+condition input的个数即实际channel输入数据的猜测的memory时间

data_dim=1000
d_input_dim=data_dim+condition_dim

batch_size=100
#g_batch_size=d_batch_size*d_input_dim
#generator
class G_mlp_hxwn1(object):
	def __init__(self):
		self.name = "G_mlp_hxwn1"
		self.X_dim = 1000

	def __call__(self, z):
		with tf.variable_scope(self.name) as vs:
			g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
		return g
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class G_mlp_hxwn2(object):
	def __init__(self):
		self.name = "G_mlp_hxwn2"
		self.X_dim = 1000

	def __call__(self, z):
		with tf.variable_scope(self.name) as vs:
			g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
		return g

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_hxwn():
	def __init__(self):
		self.name = "D_mlp_hxwn"

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
			
		return d, q

	@property
	def vars(self):		
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
def generator3(g_input):
	#g_input=tf.placeholder(tf.float32, shape=[None, g_input_dim])
	g_dense1=tcl.fully_connected(g_input,128,activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0,0.02))
	g_dense2=tcl.fully_connected(g_dense1,128,activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0,0.02))
	g_dense3=tcl.fully_connected(g_dense2,128,activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0,0.02))
	g=tcl.fully_connected(g_dense3,data_dim,activation_fn=None,weights_initializer=tf.random_normal_initializer(0,0.02))
	return g

#discriminator
def discriminator3(d_input):
	#d_input=tf.placeholder(tf.float32,shape=[None,d_input_dim])
	d_dense1=tcl.fully_connected(d_input,32,activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0,0.02))
	d_dense2=tcl.fully_connected(d_dense1,32,activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0,0.02))
	d_dense3=tcl.fully_connected(d_dense2,32,activation_fn=tf.nn.relu,weights_initializer=tf.random_normal_initializer(0,0.02))
	d=tcl.fully_connected(d_dense3,1,activation_fn=None,weights_initializer=tf.random_normal_initializer(0,0.02))
	return d
#net
generator1=G_mlp_hxwn1()
generator2=G_mlp_hxwn2()
discriminator=D_mlp_hxwn()

g_x_input_small=tf.placeholder(tf.float32,shape=[batch_size,condition_dim])
g_x_input=tf.placeholder(tf.float32,shape=[batch_size,data_dim])
g_h_input=tf.placeholder(tf.float32, shape=[batch_size, g_input_dim])#None会在后面被feed一个batchsize
g_h_output=generator1(g_h_input)#输出g_output的shape是[batchsize,data_dim]
g_hx_output=tf.multiply(g_x_input,g_h_output)

g_n_input=tf.placeholder(tf.float32,shape=[batch_size,g_input_dim])
g_n_output=generator2(g_n_input)

g_output=g_hx_output+g_n_output
g_output_sorted,_=tf.nn.top_k(g_output,data_dim)
d_input_fake=tf.concat([g_output_sorted,g_x_input_small],1)

d_input_real=tf.placeholder(tf.float32,shape=[batch_size,d_input_dim])
d_fake_output,_=discriminator(d_input_fake)
d_real_output,_=discriminator(d_input_real,reuse=True)
#loss
g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_output,labels=tf.ones_like(d_fake_output)))
d_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_output,labels=tf.ones_like(d_real_output)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_output,labels=tf.zeros_like(d_fake_output)))

#solver
d_solver=tf.train.AdamOptimizer().minimize(d_loss,var_list=discriminator.vars)
g_solver=tf.train.AdamOptimizer().minimize(g_loss,var_list=generator1.vars+generator2.vars)

#training
training_epoches=100000
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

def channel_input(batch_size,data_dim,condition_dim):
	path='E:/project_communication/hxwInput.mat'
	#path='D:/学习/Imperial/hxwInput.mat'
	data=sio.loadmat(path)
	data=data['input']
	rows=np.random.randint(data.shape[0],size=batch_size)
	data_output=data[rows]
	data_output=data_output[:,0:data_dim]
	data_output_small=data_output[:,0:condition_dim]
	# for i in range(condition_dim):
	# 	data=np.append(data[-i-1],data)
	# data_output=data[rows].reshape(batch_size,1)
	# r=rows
	# for j in range(condition_dim-1):
	# 	r=np.array([x+1 for x in r])
	# 	data_output=np.append(data[r].reshape(batch_size,1),data_output,axis=1)
	return data_output_small,data_output,rows

def channel_output(data_dim,index):
	path='E:/project_communication/hxwOutput.mat'
	data=sio.loadmat(path)
	data=data['output']
	#rows=np.random.randint(data.shape[0],size=batch_size)
	cols=np.random.randint(data.shape[1],size=data_dim)
	data=data[index]
	data=data[:,cols]
	return data

def random_generate(batch_size,g_random_dim):#返回G的输入，即一个white noize, size为[batch_size,g_random_dim]
	return np.random.uniform(-1., 1., size=[batch_size,g_random_dim]).astype(np.float32)

for epoch in range(training_epoches):
	z_h=random_generate(batch_size,g_random_dim)#生成g_batch_size*g_random_dim的random number
	z_wn=random_generate(batch_size,g_random_dim)
	y_small,y,index=channel_input(batch_size,data_dim,condition_dim)#在channel的输入中随机选的g_batch_size组condition,每组是condition_dim个时间连续的input.index是这组condition最后一个数对应的时间,是一个g_batch_size维的列向量
	g_h_in=z_h
	g_wn_in=z_wn

	x=channel_output(data_dim,index)#原channel的输出数据文件为一个矩阵，每一行对应同一组（memory）输入.channel_output()作用是从这个矩阵中选出index对应的行数，每一行随机选出1000个数，组成一个bach_size*1000的矩阵
	#i=reshape(index,[d_batch_size,d_input_dim])对应i的每一个元素取出channel output，则其每一行都与d的这一行的fake输入对应
	x.sort(axis=1)
	
	d_in_real=np.append(x,y_small,axis=1)
	sess.run(
		d_solver,
		feed_dict={g_x_input_small:y_small,g_x_input:y,g_h_input:g_h_in,g_n_input:g_wn_in,d_input_real:d_in_real}
		)
	sess.run(
		g_solver,
		feed_dict={g_x_input_small:y_small,g_x_input:y,g_h_input:g_h_in,g_n_input:g_wn_in}
		)

	if epoch %100==0 or epoch<100:
		D_loss_curr=sess.run(
			d_loss,
			feed_dict={g_x_input_small:y_small,g_x_input:y,g_h_input:g_h_in,g_n_input:g_wn_in,d_input_real:d_in_real}
			)
		G_loss_curr=sess.run(
			g_loss,
			feed_dict={g_x_input_small:y_small,g_x_input:y,g_h_input:g_h_in,g_n_input:g_wn_in}
			)
		print('Iter:%f;D_Loss: %f;G_Loss: %f'%(epoch,D_loss_curr,G_loss_curr))

	if epoch%1000==0:
		generated_samples=sess.run(
			g_output,
			feed_dict={g_x_input_small:y_small,g_x_input:y,g_h_input:g_h_in,g_n_input:g_wn_in})
		real_samples=x[0,:]
		#画频率分布直方图
		fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
		#第二个参数是柱子宽一些还是窄一些，越大越窄越密  
		ax0.hist(generated_samples[0,:],50,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
		##pdf概率分布图，一千个数落在某个区间内的数有多少个  
		ax0.set_title('pdf_fake')
		ax1.hist(real_samples,50,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
		##pdf概率分布图，一千个数落在某个区间内的数有多少个  
		ax1.set_title('pdf_real')
		fig.subplots_adjust(hspace=0.4)  
		plt.show()



