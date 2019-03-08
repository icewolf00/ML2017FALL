import sys	
import numpy as np
import time
import subprocess

fp = [open(sys.argv[i],'r') for i in range(2,len(sys.argv))]

fp2 = open(sys.argv[1], 'w')
fp2.write('id,ans\n')

for j in range(len(fp)):
	fp[j].readline()

for i in range(5060):
	tmp = np.array([0 for i in range(6)])
	# tmp[np.random.choice([0,0])] += 1
	for j in range(len(fp)):
		a = int(fp[j].readline().split(',')[1])
		tmp[a] += 1
	max_value = 0
	for m in range(len(tmp)):
		if tmp[m] > max_value:
			max_value = tmp[m]
	candicate = []
	for k in range(len(tmp)):
		if tmp[k] == max_value:
			candicate.append(k)
	fp2.write('%d,%d\n' % (i+1,np.argmax(tmp)))
	# fp2.write('%d,%d\n' % (i+1,np.random.choice(candicate)))
