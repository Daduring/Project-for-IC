import scipy.io as sio
import numpy as np
# from pylab import *

path='E:/project_communication/h0wOutput.mat'
path2='E:/project_communication/w11norm.mat'
h=sio.loadmat(path)
w=sio.loadmat(path2)
print(h['output'].shape)#(1000, 3000)
print(w['w'].shape)
print(type(data))
print(data.keys())
xx=data['input0']
print(type(xx))
print(xx.shape)#(10000, 1)
print(np.random.randint(10,size=10))
print(type(xx[1,0]))
# plot(xx)
# shiw()