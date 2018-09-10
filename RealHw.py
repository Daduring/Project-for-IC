#真实的h w中选10000个数画出hist分布图

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from pylab import *

path='E:/project_communication/h101.mat'
path2='E:/project_communication/w101.mat'
data_h=sio.loadmat(path)
data_w=sio.loadmat(path2)
h=data_h['h'].reshape(600,10000)
w=data_w['w'].reshape(600,10000)


fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
#第二个参数是柱子宽一些还是窄一些，越大越窄越密  
#print(generated_h_samples[0,:])
n, bins, patches =ax0.hist(h[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
##pdf概率分布图，一千个数落在某个区间内的数有多少个  
ax0.set_title('h')
ax1.hist(w[0,:],50,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
##pdf概率分布图，一千个数落在某个区间内的数有多少个  
ax1.set_title('w')
fig.subplots_adjust(hspace=0.4)  
plt.show()