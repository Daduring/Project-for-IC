import numpy as np
a = np.arange(0,12,0.5).reshape(4,-1)
np.savetxt("a.txt", a) # 缺省按照'%.18e'格式保存数据，以空格分隔
b=np.loadtxt("a.txt")
print(b)
np.savetxt("a.txt", a, fmt="%d", delimiter=",") #改为保存为整数，以逗号分隔
np.loadtxt("a.txt",delimiter=",") # 读入的时候也需要指定逗号分隔