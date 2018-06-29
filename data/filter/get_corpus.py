from __future__ import print_function
import argparse
import pprint
import gensim

from glove import Glove
from glove import Corpus

if __name__ == '__main__':
	texts = []
	for line in open("bag_of_words"):
		text = [item[2:-1] for item in line.strip()[1:-1].split(", ")]
		texts.append(text)
	corpus_model = Corpus()
	corpus_model.fit(texts, window=10)
	corpus_model.save('corpus.model')
	print('Dict size: %s' % len(corpus_model.dictionary))
	print('Collocations: %s' % corpus_model.matrix.nnz)
