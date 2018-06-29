from __future__ import print_function
import argparse
import pprint
import gensim

from glove import Glove
from glove import Corpus

if __name__ == '__main__':
	print('Reading corpus statistics')
	corpus_model = Corpus.load('corpus.model')
	print('Dict size: %s' % len(corpus_model.dictionary)) 
	print('Collocations: %s' % corpus_model.matrix.nnz)

	print('Training the GloVe model')
	glove = Glove(no_components=100, learning_rate=0.05)
	glove.fit(corpus_model.matrix, epochs=10, verbose=True)
	glove.add_dictionary(corpus_model.dictionary)
	glove.save('glove.model')
	print('Training finished') 
	
