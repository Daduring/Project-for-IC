import os,sys
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio

from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'
def get_img(img_path, crop_h, resize_h):
	img=scipy.misc.imread(img_path).astype(np.float)
	# crop resize
	crop_w = crop_h
	#resize_h = 64
	resize_w = resize_h
	h, w = img.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])

	return np.array(cropped_image)/255.0

class face3D():
	def __init__(self):
		datapath = '/ssd/fengyao/pose/pose/images'
		self.z_dim = 100
		self.c_dim = 2
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-1:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, 256, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		#fig = self.data2fig(batch_imgs[:16,:,:])
		#plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
		#plt.close(fig)
		
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class celebA():
	def __init__(self):
		datapath = prefix + 'celebA'
		self.z_dim = 100
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-1:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		'''
		print self.batch_count
		fig = self.data2fig(batch_imgs[:16,:,:])
		plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
		plt.close(fig)
		'''
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class mnist():
	def __init__(self, flag='conv', is_tanh = False):
		datapath = prefix + 'mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10
		self.size = 28 # for conv
		self.channel = 1 # for conv
		self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):
		batch_imgs,y = self.data.train.next_batch(batch_size)
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2-1		
		return batch_imgs, y

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples+1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig

class channel_hx_plus_wn():
	def __init__(self, flag='conv', is_tanh = False):

		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10#10个相同的数，等于输入，重复十遍作用是加强其对结果的影响
		self.size = 28 # for conv
		self.channel = 1 # for conv
		# self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):#X_b,y_b,y_stack_b = self.data(batch_size)
		# batch_imgs,y = self.data.train.next_batch(batch_size)
		path1 = prefix + 'channelData/h101w101input.mat'
		path2= prefix + 'channelData/h101w101output.mat'
		#path1='E:/project_communication/UNIWNInput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path1)
		data=data['input']
		rows=np.random.randint(data.shape[0],size=batch_size)
		channel_input=data[rows]
		channel_input_condition=channel_input[:,0:self.y_dim]
		channel_input_stack=channel_input[:,0:self.X_dim]

		path2='E:/project_communication/UNIWNOutput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path2)
		data=data['output']
		channel_output=data[rows]
		cols=np.random.randint(data.shape[0],size=self.X_dim)
		channel_output=channel_output[:,cols]
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return channel_output, channel_input_condition,channel_input_stack

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig	
class channel_hdx_plus_wn():
	def __init__(self, flag='conv', is_tanh = False):
		# datapath = prefix + 'mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 11#one-hot
		self.size = 28 # for conv
		self.channel = 1 # for conv
		# self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):#X_b,y_b,y_stack_b = self.data(batch_size)
		# batch_imgs,y = self.data.train.next_batch(batch_size)
		path1='E:/project_communication/2hdx1wInput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path1)
		data=data['input']
		rows=np.random.randint(data.shape[0],size=batch_size)
		channel_input=data[rows]
		channel_input_condition=np.zeros([channel_input.shape[0],self.y_dim])
		cols2=channel_input[:,0]
		i=0
		for col in cols2:
			# print (int(col))
			channel_input_condition[i,int(col)]=1
			i=i+1
		channel_input_stack=channel_input[:,0:self.X_dim]

		path2='E:/project_communication/2hdx1wOutput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path2)
		data=data['output']
		channel_output=data[rows]
		cols=np.random.randint(data.shape[0],size=self.X_dim)
		channel_output=channel_output[:,cols]
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return channel_output, channel_input_condition,channel_input_stack

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig	

class channel_h10x_plus_wn():
	def __init__(self, flag='conv', is_tanh = False):
		# datapath = prefix + 'mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10#repeated
		self.size = 28 # for conv
		self.channel = 1 # for conv
		# self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):#X_b,y_b,y_stack_b = self.data(batch_size)
		# batch_imgs,y = self.data.train.next_batch(batch_size)
		path1='E:/project_communication/21h11wInput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path1)
		data=data['input']
		rows=np.random.randint(data.shape[0],size=batch_size)
		channel_input=data[rows]
		channel_input_condition=channel_input[:,0:self.y_dim]
		# cols2=channel_input[:,0]
		# i=0
		# for col in cols2:
		# 	# print (int(col))
		# 	channel_input_condition[i,int(col)]=1
		# 	i=i+1
		channel_input_stack=channel_input[:,0:self.X_dim]

		path2='E:/project_communication/21h11wOutput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path2)
		data=data['output']
		channel_output=data[rows]
		cols=np.random.randint(data.shape[0],size=self.X_dim)
		channel_output=channel_output[:,cols]
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return channel_output, channel_input_condition,channel_input_stack

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig	

class channel_with_memory():
	def __init__(self, flag='conv', is_tanh = False):
		# datapath = prefix + 'mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.memory_dim = 5
		self.size = 28 # for conv
		self.channel = 1 # for conv
		# self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):#X_b,y_b,y_stack_b = self.data(batch_size)
		# batch_imgs,y = self.data.train.next_batch(batch_size)
		path1='E:/project_communication/memory1_2_1.5Input.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path1)
		data=data['input0']
		rows=np.random.randint(data.shape[0],size=batch_size)
		for i in range(self.memory_dim):
			data=np.append(data[-i-1],data)
		data_output=data[rows].reshape(batch_size,1)

		r=rows

		for j in range(self.memory_dim-1):
			r=np.array([x+1 for x in r])
			data_output=np.append(data[r].reshape(batch_size,1),data_output,axis=1)

		y_stack=data_output
		y0=np.tile(y_stack[:,0].reshape(batch_size,1),self.X_dim)
		y1=np.tile(y_stack[:,1].reshape(batch_size,1),self.X_dim)
		y2=np.tile(y_stack[:,2].reshape(batch_size,1),self.X_dim)
		y3=np.tile(y_stack[:,3].reshape(batch_size,1),self.X_dim)
		y4=np.tile(y_stack[:,4].reshape(batch_size,1),self.X_dim)

		path2='E:/project_communication/memory1_2_1.5Output.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path2)
		data=data['output']
		channel_output=data[rows]
		cols=np.random.randint(data.shape[1],size=self.X_dim)
		X=channel_output[:,cols]
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return X, y0, y1, y2, y3, y4, y_stack

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig

class channel_21h11w_plus_wn_shape500_20000():
	def __init__(self, flag='conv', is_tanh = False):
		# datapath = prefix + 'mnist'
		self.X_dim = 3000 # for mlp
		self.z_dim = 100
		self.y_dim = 30#repeated
		self.size = 28 # for conv
		self.channel = 1 # for conv
		# self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):#X_b,y_b,y_stack_b = self.data(batch_size)
		# batch_imgs,y = self.data.train.next_batch(batch_size)
		path1='E:/project_communication/21h11wInput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path1)
		data=data['input']
		rows=np.random.randint(data.shape[0],size=batch_size)
		channel_input=data[rows]
		channel_input_condition=channel_input[:,0:self.y_dim]
		# cols2=channel_input[:,0]
		# i=0
		# for col in cols2:
		# 	# print (int(col))
		# 	channel_input_condition[i,int(col)]=1
		# 	i=i+1
		channel_input_stack=channel_input[:,0:self.X_dim]

		path2='E:/project_communication/21h11wOutput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path2)
		data=data['output']
		channel_output=data[rows]
		cols=np.random.randint(data.shape[0],size=self.X_dim)
		channel_output=channel_output[:,cols]
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return channel_output, channel_input_condition,channel_input_stack

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig

class channel_hx0_plus_wn():
	def __init__(self, flag='conv', is_tanh = False):
		# datapath = prefix + 'mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10#10个相同的数，等于输入，重复十遍作用是加强其对结果的影响
		self.size = 28 # for conv
		self.channel = 1 # for conv
		# self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):#X_b,y_b,y_stack_b = self.data(batch_size)
		# batch_imgs,y = self.data.train.next_batch(batch_size)
		path1='E:/project_communication/h0x1wInput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path1)
		data=data['input']
		rows=np.random.randint(data.shape[0],size=batch_size)
		channel_input=data[rows]
		channel_input_condition=channel_input[:,0:self.y_dim]
		channel_input_stack=channel_input[:,0:self.X_dim]

		path2='E:/project_communication/h0x1wOutput.mat'
		#path='D:/学习/Imperial/hxwInput.mat'
		data=sio.loadmat(path2)
		data=data['output']
		channel_output=data[rows]
		cols=np.random.randint(data.shape[0],size=self.X_dim)
		channel_output=channel_output[:,cols]
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return channel_output, channel_input_condition,channel_input_stack

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig