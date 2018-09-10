import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

path='E:/project_communication/GAN-master/GAN-master/results/wgan/generated_h.txt'
path2='E:/project_communication/GAN-master/GAN-master/results/wgan/generated_h.txt'
path5='E:/project_communication/GAN-master/GAN-master/results/generated_h.txt'
path6='E:/project_communication/GAN-master/GAN-master/results/generated_w.txt'
path3='E:/project_communication/GAN-master/GAN-master/results/h101w101ordinary/generated_h.txt'
path4='E:/project_communication/GAN-master/GAN-master/results/h101w101ordinary/generated_w.txt'
h_gan=np.loadtxt(path5)
w_gan=np.loadtxt(path6)

h=np.loadtxt(path)
w=np.loadtxt(path2)
oh=np.loadtxt(path3)
ow=np.loadtxt(path4)

num_bins=50
hmean=1
hstd=0.1
wmean=1
wstd=0.1

plt.figure(num='astronaut',figsize=(10,15))

plt.subplot(3,2,1)
plt.title('h generated using adversarial architecture')
n1, bins1, patches1 = plt.hist(h[35,:]-0.4, num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins1, hmean, hstd)  
plt.plot(bins1, y, 'r--')  
plt.xlabel('h')  
plt.ylabel('Frequency Density')  

plt.subplot(3,2,2)
plt.title('w generated using adversarial architecture')
n2, bins2, patches2 = plt.hist(w[40,:]-0.65, num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins2, wmean, wstd)  
plt.plot(bins2, y, 'r--')  
plt.xlabel('w')  
plt.ylabel('Frequency Density') 

plt.subplot(3,2,3)
plt.title('h generated using ordinary loss')
n2, bins2, patches2 = plt.hist(oh[24,:], num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins2, wmean, wstd)
plt.plot(bins2, y, 'r--')  
plt.xlabel('w')  
plt.ylabel('Frequency Density')

plt.subplot(3,2,4)
plt.title('w generated using ordinary loss')
n2, bins2, patches2 = plt.hist(ow[25,:], num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins2, wmean, wstd)
plt.plot(bins2, y, 'r--')  
plt.xlabel('w')  
plt.ylabel('Frequency Density')

plt.subplot(3,2,5)
plt.title('h generated using existing method')
n2, bins2, patches2 = plt.hist(h_gan[24,:], num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins2, wmean, wstd)
 
plt.xlabel('w')  
plt.ylabel('Frequency Density')

plt.subplot(3,2,6)
plt.title('w generated using existing method')
n2, bins2, patches2 = plt.hist(w_gan[25,:], num_bins, normed=1, facecolor='blue', alpha=0.5)
y = mlab.normpdf(bins2, wmean, wstd)
 
plt.xlabel('w')  
plt.ylabel('Frequency Density')
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

plt.savefig("wgan_complexW1_01.png")
plt.show() 