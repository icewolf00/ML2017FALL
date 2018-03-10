import pandas as pd
import numpy as np
import sys

# train_csv = sys.argv[1]
# test_csv = sys.argv[2]
# ans_csv = sys.argv[3]

def cut_train(train_data):
	put_in = []
	throw_out = []
	put_in_mix = []
	(h, w) = train_data.shape		#4320, 24
	#print(type(h))
	for i in range(0, h, int(h/12)):
		if i == 2160:
			continue
		#20array: 18row(type), 24col(hour)
		day_block = np.vsplit(train_data[i:i+int(h/12)], 20)
		#18row(types), 480col(hours_mix)
		month_block = np.concatenate(day_block, axis = 1)
		#month_block[7,:] = month_block[2,:] * month_block[7,:]
		#print(month_block.shape[1]-9)
		for j in range(0, month_block.shape[1] - 9):			#month_block.shape[1] - 9 = 471
			#put_in.append([month_block[:, j : j + 9]])
			put_in.append(month_block[9, j : j + 9])
			put_in_mix.append(month_block[10, j : j + 9])
			throw_out.append(month_block[9, j + 9])
	#print(len(put_in))			#5652 = 12 * (471)
	#print(len(put_in[0]))		#18
	#print(len(put_in[0][0]))	#9
	put_in_mix = np.array(put_in_mix)
	put_in = np.array(put_in)
	throw_out = np.array(throw_out)
	return put_in, put_in_mix, throw_out

def cut_test(test_data):
	test_block = []
	test_block_mix = []
	(h, w) = test_data.shape
	#240array: 18row(type), 9col(hour)
	temp_block = np.array(np.vsplit(test_data, int(h/18)))
	for i in range(len(temp_block)):
		temp_block[i][7] = temp_block[i][7] * temp_block[i][2]
		test_block.append(temp_block[i][9])
		test_block_mix.append(temp_block[i][10])
	#print(test_block[0:2])
	test_block = np.array(test_block)
	test_block_mix = np.array(test_block_mix)
	return test_block, test_block_mix


def readtrain(file_name):
	file_train = pd.read_csv(file_name, encoding = "big5")
	file_train_list = file_train.as_matrix()
	data = file_train_list[:,3:]
	data[data == "NR"] = 0.0
	data = data.astype(float)
	return cut_train(data)

def readtest(file_name):
	file_test = pd.read_csv(file_name, encoding = "big5", header = None)
	file_test_list = file_test.as_matrix()
	#print(file_test_list)
	data = file_test_list[:,2:]
	data[data == "NR"] = 0.0
	data = data.astype(float)
	return cut_test(data)

def runtrain(train_in, train_in_mix, train_out):
	b = -0.1
	w = np.array([0.5] * 9)
	w2 = np.array([0.5] * 9)
	wmix = np.array([0.5] * 9)
	lr = 0.5
	iteration = 5000
	b_lr = 0.1
	w_lr = np.array([0.1] * 9)
	w_lr2 = np.array([0.1] * 9)
	w_lrmix = np.array([0.1] * 9)
	length = len(train_in)
	sum_multi = 0
	last_multi = 0
	train_in2 = train_in ** 2
	#sum_array = np.array([train_in[i].sum() for i in range(len(train_in))])
	#for i in range(iteration):
	while 1:
		b_grad = 0.0
		w_grad = np.array([0.0] * 9)
		w_grad2 = np.array([0.0] * 9)
		w_gradmix = np.array([0.0] * 9)
		sum_multi = 0.0
		#print(i)
		temp_multi = np.array(train_in.dot(np.transpose(w)))
		temp_multi2 = np.array(train_in2.dot(np.transpose(w2)))
		temp_multimix = np.array(train_in_mix.dot(np.transpose(wmix)))
		sum_multi = (((train_out - b - temp_multi - temp_multi2 - temp_multimix) **2).sum()/length)**0.5
		print(sum_multi)
		if abs(sum_multi - last_multi) < (5* 1e-7):
			break
		last_multi = sum_multi
		#print(((train_out - b - temp_multi)*sum_array).shape)
		w_grad = w_grad - (2.0*(np.transpose(train_in).dot(train_out - b - temp_multi - temp_multi2 - temp_multimix)))/length
		w_grad2 = w_grad2 - (2.0*(np.transpose(train_in2).dot(train_out - b - temp_multi - temp_multi2 - temp_multimix)))/length
		w_gradmix = w_gradmix - (2.0*(np.transpose(train_in_mix).dot(train_out - b - temp_multi - temp_multi2 - temp_multimix)))/length
		b_grad = b_grad - (2.0*((train_out - b - temp_multi - temp_multi2 - temp_multimix).sum()))/length

		b_lr = b_lr + b_grad**2
		w_lr = w_lr + w_grad**2
		w_lr2 = w_lr2 + w_grad2**2
		w_lrmix = w_lrmix + w_gradmix**2
		w = w - lr/np.sqrt(w_lr) * w_grad
		w2 = w2 - lr/np.sqrt(w_lr2) * w_grad2
		wmix = wmix - lr/np.sqrt(w_lrmix) * w_gradmix
		b = b - lr/np.sqrt(b_lr) * b_grad
		
	return b, w, w2, wmix

def runtest(test_in, test_in_mix, b, w, w2, wmix, ans_csv):
	file_out = open(ans_csv, "w")
	file_out.write("id,value\n")
	test_in2 = test_in ** 2
	for n in range(len(test_in)):
		temp_multi = b + test_in[n].transpose().dot(w) + test_in2[n].transpose().dot(w2) + test_in_mix[n].transpose().dot(wmix)
		temp_str = "id_" + str(n) + "," +  str(temp_multi) + "\n"
		file_out.write(temp_str)

def minnortrain(train_in):
	in_min = np.min(train_in, axis = 0)
	in_max = np.max(train_in, axis = 0)
	train_in = (train_in - in_min) / (in_max - in_min)
	return train_in, in_max, in_min

def minnortest(test_in, in_max, in_min):
	test_in = (test_in - in_min) / (in_max - in_min)
	return test_in

def minmixtrain(train_in_mix):
	mix_min = np.min(train_in_mix, axis = 0)
	mix_max = np.max(train_in_mix, axis = 0)
	train_in_mix = (train_in_mix - mix_min) / (mix_max - mix_min)
	return train_in_mix, mix_max, mix_min

def minmixtest(test_in_mix, mix_max, mix_min):
	test_in_mix = (test_in_mix - mix_min) / (mix_max - mix_min)
	return test_in_mix

# train_csv = sys.argv[1]
test_csv = sys.argv[1]
ans_csv = sys.argv[2]
w = []
w2 = []
wmix = []
in_max = []
in_min = []
mix_max = []
mix_min = []
file_model = open("model", "r")
for i in range(9):
	w.append(float(file_model.readline()))
w = np.array(w)
for i in range(9):
	w2.append(float(file_model.readline()))
w2 = np.array(w2)
for i in range(9):
	wmix.append(float(file_model.readline()))
wmix = np.array(wmix)
b = float(file_model.readline())
for i in range(9):
	in_max.append(float(file_model.readline()))
in_max = np.array(in_max)
for i in range(9):
	in_min.append(float(file_model.readline()))
in_min = np.array(in_min)
for i in range(9):
	mix_max.append(float(file_model.readline()))
mix_max = np.array(mix_max)
for i in range(9):
	mix_min.append(float(file_model.readline()))
mix_min = np.array(mix_min)

# train_in = []
# train_out = []
# (train_in, train_in_mix, train_out) = readtrain("train.csv")
# (train_in, in_max, in_min) = minnortrain(train_in)
# (train_in_mix, mix_max, mix_min) = minmixtrain(train_in_mix)
#(b, w, w2, wmix) = runtrain(train_in, train_in_mix, train_out)
test_in = []
(test_in, test_in_mix) = readtest(test_csv)
test_in = minnortest(test_in, in_max, in_min)
test_in_mix = minmixtest(test_in_mix, mix_max, mix_min)
runtest(test_in, test_in_mix, b, w, w2, wmix, ans_csv)