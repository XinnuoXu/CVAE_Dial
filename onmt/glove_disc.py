#coding=utf8

import sys
import math
import torch
import gensim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import cosine

from glove import Glove
from glove import Corpus

class GloVe_Discriminator(object):
	def __init__(self, cuda):
		self.cuda = cuda
		self.tt = torch.cuda if cuda else torch
		self.dict_path = "./data/dialogue.vocab.pt"
		self.no_components = 100
		self.learning_rate = 0.05
		# Corpus model
		self.vocab = dict(torch.load(self.dict_path, "text")) 
		self.corpus_model = Corpus(dictionary=self.vocab['tgt'].stoi)
		# Model
		self.glove = Glove(no_components=self.no_components, learning_rate=self.learning_rate)

	def load_model(self, model_path):
		print('Loading pre-trained GloVe model...')
		self.glove = Glove.load(model_path)
		print('Loading finished')

	def get_emb(self, ids):
		emb = self.glove.word_vectors[ids.data.cpu().numpy()]
		paragraph_emb = np.mean(emb, axis=0)
		return paragraph_emb

	def run(self, src, tgt):
		src = src.view(src.size()[0], -1)
		tgt = tgt.view(tgt.size()[0], -1)
		src_emb = self.get_emb(src)
		tgt_emb = self.get_emb(tgt)
		return [1 - cosine(src_emb[i], tgt_emb[i]) for i in range(0, src_emb.shape[0])]

	def run_iter(self, src, tgt):
		src = src.view(src.size()[0], -1)
		src_emb = self.get_emb(src)
		tgt = tgt.view(tgt.size()[0], -1)
		sim_list = []
		for i in range(0, tgt.size()[0]):
		    tgt_emb = self.get_emb(tgt[:i+1, :].view(-1, tgt.size()[1]))
		    sim_list.append([1 - cosine(src_emb[j], tgt_emb[j]) for j in range(0, src_emb.shape[0])])
	        return np.array(sim_list).transpose()

	def run_soft(self, src, tgt):
		# Src emb
		batch_size = src.shape[1]
		src = src.view(src.size()[0], -1)
		src_emb = Variable(self.tt.FloatTensor(self.get_emb(src)))

		# Soft tgt emb
		word_emb = Variable(self.tt.FloatTensor(self.glove.word_vectors))
		#tgt_emb = sum([torch.mm(item, word_emb) for item in tgt]).data.cpu().numpy()
		tgt_emb = sum([torch.mm(item, word_emb) for item in tgt])

		# similarity
		return F.cosine_similarity(src_emb, tgt_emb, dim=1)
