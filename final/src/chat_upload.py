import sys
import pandas as pd
import numpy as np
import jieba.analyse
from gensim.models import Word2Vec
import jieba
from jieba import analyse
from scipy import spatial
jieba.set_dictionary("../dict.txt.big")
analyse.set_stop_words("../stop.txt")
model = Word2Vec.load(sys.argv[1])
word2idx = {"_PAD": 0}
#vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
# embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
# print(model.similarity(,))

sum_all = np.sum(model.wv.vocab[i].count for i in model.wv.vocab)
def parseData(data):
	return [ (sent.replace('A:','')).replace('B:','') for sent in data]

def jiebaSeg(lines):
	segLine = []
	words = jieba.cut(lines)
	for word in words:
		if word != ' ' and word != '':
			segLine.append(word)
	return segLine

def sim(quest,opt):
	quest = quest.split()
	opt = opt.split('\t')


	max_sim = 0
	max_sim2 = 0
	opt_len = len(opt)

	questSeg = []
	optSeg = []

	tmp_quest = np.zeros(model.vector_size)
	tmp_option = np.zeros((6,model.vector_size))
	for q in quest:
		qSeg = jiebaSeg(q)
		questSeg += [i for i in qSeg]
	for o in opt:
		oSeg = jiebaSeg(o)
		if oSeg is None:
			continue
		optSeg.append(oSeg)
	sum_sim = np.zeros((opt_len, 1), dtype = float)
	threshold = 0.29

	for i in questSeg:
		if i in model.wv:
			tmp_quest += (model.wv[i]) * (1e-3  / (1e-3 + (model.wv.vocab[i].count / sum_all)))
	tmp_quest /= len(tmp_quest)
	for index,i in enumerate(optSeg):
		all_zero = 1
		for j in i:
			if j in model.wv:
				all_zero = 0
				tmp_option[index] += model.wv[j] * (1e-3  / (1e-3 + (model.wv.vocab[j].count / sum_all)))
		tmp_option[index] /= len(tmp_option)
		if all_zero == 0:	
			sum_sim[index] = 1 - (spatial.distance.cosine(tmp_option[index],tmp_quest))
	#sum_sim = (tmp_option.dot(tmp_quest))
	sum_sim /= sum_sim.max()
	#sum_sim[0] * 1.126
	#sum_sim *= 45
	sum_sim *= 8.88 * 1.126 * 5.01 * 1.337 
	#return sum_sim
	for i, one_opt in enumerate(optSeg):
		for k, dialog_seg in enumerate(questSeg):
			for opt_seq in one_opt:
				try:
					sim = abs(model.similarity(dialog_seg,opt_seq))
					if sim > threshold:
						sum_sim[i] = sum_sim[i] + sim
				except Exception as e:
					pass
	sum_sim[0] *= 1.1
	return sum_sim

test_path = sys.argv[3]
test_data = pd.read_csv(test_path)

test_data['dialogue'] = parseData(test_data['dialogue'])
test_data['options']  = parseData(test_data['options'])

with open(sys.argv[2],'w') as fp:
	print('id,ans',file = fp)
	for i in range(len(test_data['dialogue'])):
		if i in [1000,2000,3000,4000,5000]:
			print(i)
		max_index = np.argmax(sim(test_data['dialogue'][i],test_data['options'][i]))
		print('%d,%d' %(i+1,max_index),file = fp)
# for i in range(len(vocab_list)):
#     word = vocab_list[i][0]
#     word2idx[word] = i + 1
#     embeddings_matrix[i + 1] = vocab_list[i][1]

