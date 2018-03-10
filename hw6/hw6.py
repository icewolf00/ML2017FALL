from keras.models import load_model
from sklearn import cluster
import numpy as np
import pandas as pd
import sys

# model = load_model("models/model-00176-0.11606.h5")
model = load_model(sys.argv[4])
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]

# model = load_model("models/encoder.h5")

X_train = np.load(sys.argv[1])
# X_train = X_train / 255
X_mean = np.mean(X_train, axis = 0)
X_std = np.std(X_train, axis = 0)
# np.save("normal.npy", [X_mean, X_std])
X_train = (X_train - X_mean) / (X_std + 1e-15)

X_encoded = model.predict(X_train, batch_size=128)

# print(X_encoded[0])

clf = cluster.KMeans(n_clusters=2)
clusters = clf.fit_predict(X_encoded)
temp = 0
for i in clusters:
    temp += i
print(temp)


X_test = pd.read_csv(sys.argv[2])
ans = []
counter = 0
for i in range(len(X_test['ID'])):
    if clusters[X_test['image1_index'][i]] == clusters[X_test['image2_index'][i]]:
        ans.append(1)
        counter += 1
    else:
        ans.append(0)
ans = np.array(ans)
print(counter)

# np.save("clusters_dnn_0417.npy", clusters)
# np.save("ans_0417.npy", ans)

file_out = open(sys.argv[3], "w")
print("ID,Ans", file = file_out)
for i in range(len(ans)):
    if i == (len(ans) - 1):
        print("end")
    file_out.write(str(i) + "," + str(ans[i]) + '\n')
#     print(str(i) + "," + str(ans[i]), file = file_out)
file_out.close()