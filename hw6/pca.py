from skimage import io
import numpy as np
import sys
import os
imgs = []
average = 0

pic_list = os.listdir(sys.argv[1])
# print(pic_list)
for i in pic_list:
    file_name = sys.argv[1] + "/" + str(i)
    img = io.imread(file_name).flatten()
    average += img
    imgs.append(img)

X_mean = np.mean(imgs, axis = 0)
imgs = np.array(imgs)

X = imgs - X_mean
X = X.T

U,s,V = np.linalg.svd(X, full_matrices=False)

U_use = U.T

#1.2
# for i in range(4):
# 	M = U_use[i].reshape(600,600,3)
# 	M -= np.min(M)
# 	M /= np.max(M)
# 	M = (M * 255).astype(np.uint8)
# 	io.imshow(M)
# 	io.show()

#1.3
file_name = sys.argv[1] + "/" + sys.argv[2]
to_constr = io.imread(file_name).flatten()
M = X_mean.reshape(600, 600, 3)
for j in range(4):
	M += np.dot(U_use[j], to_constr - X_mean) * U_use[j].reshape(600, 600, 3)
M -= np.min(M)
M /= np.max(M)
M = (M * 255).astype(np.uint8)
io.imsave("reconstruction.jpg", M, quality = 100)

#1.4
# print(s)
# total_s = np.sum(s)
# for i in range(4):
# 	print(s[i] / total_s)