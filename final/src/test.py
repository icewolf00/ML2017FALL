import os
import sys
import subprocess

test_path = sys.argv[1]
output_path = sys.argv[2]

with open('../used','r') as fp:
	for index,line in enumerate(fp):
		index = index + 1
		line = line.strip('\r\n').replace('\\','')
		line = os.path.join('../models',line)
		print("python3 chat_upload.py " + line + " " + str(index) + ".csv " + test_path)
		subprocess.call("python3 chat_upload.py " + line + " " + str(index) + ".csv " + test_path, shell = True)
	subprocess.call("python3 model_merge.py " + output_path + " 1.csv 2.csv 2.csv 3.csv 4.csv 4.csv 5.csv 6.csv 7.csv 8.csv 8.csv 9.csv ", shell = True)