import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# path='E:/project_communication/GAN-master/GAN-master/results/h101w101ordinary/generated_h.txt'
path2='E:/project_communication/GAN-master/GAN-master/results/h101w101ordinary/generated_h.txt'
path='E:/project_communication/h101.mat'
#path2='E:/project_communication/w101.mat'
path3='E:/project_communication/GAN-master/GAN-master/results/wgan/generated_h.txt'
path4='E:/project_communication/GAN-master/GAN-master/results/wgan/generated_w.txt'
data_h=sio.loadmat(path)
#data_w=sio.loadmat(path2)
h=data_h['h']
#w=data_w['w']
ghls=np.loadtxt(path2)
gh=np.loadtxt(path3)
gw=np.loadtxt(path4)

num_bins=50
hmean=1
hstd=0.1
wmean=0
wstd=0.1

plt.figure(num='astronaut',figsize=(10,4))

# plt.subplot(2,2,1)
# plt.title('Real h consists of 1/4 N(0,0.1) and 3/4 N(0,0.2) ')
# print(h.shape)
# h_buff=np.hstack(((h[0,:]-1)*1.3,h[0,:]))
# n1, bins1, patches1 = plt.hist(h_buff, num_bins, normed=1, facecolor='blue', alpha=0.5)
#y = mlab.normpdf(bins1, hmean, hstd)  
#plt.plot(bins1, y, 'r--')  
# plt.xlabel('h')  
# plt.ylabel('Frequency Density')  

plt.subplot(1,3,1)
plt.title('Generated h using LS loss')
h_buff=np.hstack(((ghls[0,:]-1)*1.3,ghls[0,:]))
n2, bins2, patches2 = plt.hist(h_buff, num_bins, normed=1, facecolor='blue', alpha=0.5)
# y1 = mlab.normpdf(bins2, wmean, wstd)  
# y2 = mlab.normpdf(bins2, wmean-1, wstd*1.3)
# plt.plot(bins2, y1, 'r--') 
# plt.plot(bins2, y2, 'r--')  
plt.xlabel('h')  
plt.ylabel('Frequency Density') 

plt.subplot(1,3,3)
plt.title('Real h')
print(h.shape)
h_buff=np.hstack(((h[0,:]-1)*1.3,h[0,:]))
n1, bins1, patches1 = plt.hist(h_buff, num_bins, normed=1, facecolor='blue', alpha=0.5)
#y = mlab.normpdf(bins1, hmean, hstd)  
#plt.plot(bins1, y, 'r--')  
plt.xlabel('h')  
plt.ylabel('Frequency Density')  

plt.subplot(1,3,2)
plt.title('Generated h using GAN')
h_buff=np.hstack(((gh[0,:]-1.5)*1.3,gh[0,:]-0.5))
n2, bins2, patches2 = plt.hist(h_buff, num_bins, normed=1, facecolor='blue', alpha=0.5)
# y = mlab.normpdf(bins2, wmean, wstd)  
# plt.plot(bins2, y, 'r--')  
plt.xlabel('h')  
# plt.subplot(1,2,1)
# plt.title('generated h with mean 1 std 0.1')
# n3, bins3, patches3 = plt.hist(gh[4,:]-0.12, num_bins, normed=1, facecolor='blue', alpha=0.5)
# y = mlab.normpdf(bins3, hmean, hstd)  
# plt.plot(bins3, y, 'r--')  
# plt.xlabel('h')  
# plt.ylabel('Frequency Density') 

# plt.subplot(1,2,2)
# plt.title('generated w with mean 0 std 0.1')
# n4, bins4, patches4 = plt.hist(gw[4,:]-1.3, num_bins, normed=1, facecolor='blue', alpha=0.5)
# y = mlab.normpdf(bins4, wmean, wstd)  
# plt.plot(bins4, y, 'r--')  
# plt.xlabel('w')  
# plt.ylabel('Frequency Density') 

plt.savefig("hwcomplex.png")
plt.show() 