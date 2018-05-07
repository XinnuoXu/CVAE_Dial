from __future__ import print_function
import sys
import math
import torch
import argparse
import pprint
import gensim
import numpy as np

from glove import Glove
from glove import Corpus
from scipy.spatial.distance import cosine
from preprocess import pre_processor


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
# Model training
parser.add_argument('--data_path', type=str, default='../../../GloVe_training_data/GloVe_train.dat',
                    help='Location of the data corpus')
parser.add_argument('--corpus_model_path', type=str, default='./corpus.model',
                    help='Location of the corpus model')
parser.add_argument('--model_path', type=str, default='./glove.model',
                    help='Location of the model')
parser.add_argument('--window', type=int, default=10,
                    help='Window size for GloVe training')
parser.add_argument('--no_components', type=int, default=100,
                    help='Training parameter')
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='Training parameter')
parser.add_argument('--epochs', type=int, default=10,
                    help='Training parameter')
# Filtering
parser.add_argument('--threshold', type=float, default=0.7,
                    help='Filtering threshold')
args = parser.parse_args()


class GloVeFilter(object):
	def __init__(self):
		# Preprocessor
		self.pp = pre_processor()
		# Corpus model
		vocab = dict(torch.load("../data/dialogue.vocab.pt", "text")) 
		self.corpus_model = Corpus(dictionary=vocab['tgt'].stoi)
		# Model
		self.glove = Glove(no_components=args.no_components, learning_rate=args.learning_rate)

	def load_corpus_from_txt(self):
		print('Reading corpus statistics...')
		#texts = [self.pp.preprocessing(l.strip().decode("utf8", "ignore")) for l in open(args.data_path)]
		texts = [l.strip().decode("utf8", "ignore").split(" ") for l in open(args.data_path)]
		self.corpus_model.fit(texts, window=args.window, ignore_missing=True)
		self.corpus_model.save(args.corpus_model_path)
		print('Dict size: %s' % len(self.corpus_model.dictionary))
		print('Collocations: %s' % self.corpus_model.matrix.nnz)

	def load_corpus_from_model(self):
		print('Reading corpus statistics...')
		self.corpus_model = Corpus.load(args.corpus_model_path)
		print('Dict size: %s' % len(self.corpus_model.dictionary)) 
		print('Collocations: %s' % self.corpus_model.matrix.nnz)

	def load_model(self):
		print('Loading pre-trained GloVe model...')
		self.glove = Glove.load(args.model_path)
		print('Loading finished')

	def train(self):
		print('Training the GloVe model...')
		self.glove.fit(self.corpus_model.matrix, epochs=args.epochs, verbose=True)
		self.glove.add_dictionary(self.corpus_model.dictionary)
		self.glove.save(args.model_path)
		print('Training finished') 
	
	def _paragraph_similarity(self, paragraph_1, paragraph_2):
		paragraph_vector_1 = self.glove.transform_paragraph(paragraph_1, ignore_missing=True)
		paragraph_vector_2 = self.glove.transform_paragraph(paragraph_2, ignore_missing=True)
		dst = (np.dot(paragraph_vector_1, paragraph_vector_2)
		       / np.linalg.norm(paragraph_vector_1)
		       / np.linalg.norm(paragraph_vector_2))
		return -1 if math.isnan(dst) else dst
	
	def filter(self, tag="test"):
		en_list = [l.strip().decode("utf8", "ignore") for l in open("../data/" + tag + ".en")]
		vi_list = [l.strip().decode("utf8", "ignore") for l in open("../data/" + tag + ".vi")]
		en_pp = [self.pp.preprocessing(l) for l in en_list]
		vi_pp = [self.pp.preprocessing(l) for l in vi_list]

		test_data = []
		for i in range(0, len(en_list)):
		    if len(en_pp[i]) != 0 and len(vi_pp[i]) != 0:
			test_data.append((en_list[i], vi_list[i], en_pp[i], vi_pp[i], 1))
		    else:
			test_data.append((en_list[i], vi_list[i], en_list[i].split(" "), vi_list[i].split(" "), 0.3))

		# To save mem, you can print out directly
		for (en, vi, enp, vip, discount) in test_data:
		    print (str(self._paragraph_similarity(enp, vip)) + "\t" + en + "\t" + vi)



if __name__ == '__main__':
	# Training
	t = GloVeFilter()
	t.load_corpus_from_txt()
	t.train()

	# Load from pre-trained model
	#t = GloVeFilter()
	#t.load_model()

	# Testing
	#t.filter("test")
