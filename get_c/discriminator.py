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


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
# Model training
parser.add_argument('--data_path', type=str, default='Disc_train.dat',
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

	def train(self):
		print('Training the GloVe model...')
		self.glove.fit(self.corpus_model.matrix, epochs=args.epochs, verbose=True)
		self.glove.add_dictionary(self.corpus_model.dictionary)
		self.glove.save(args.model_path)
		print('Training finished') 
	
if __name__ == '__main__':
	# Training
	t = GloVeFilter()
	t.load_corpus_from_txt()
	t.train()
