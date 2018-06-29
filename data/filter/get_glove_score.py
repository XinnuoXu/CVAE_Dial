#coding=utf8
from __future__ import print_function
import argparse
import pprint
import gensim

import os, sys
from glove_test import Glove
from glove import Corpus

import re
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words

class pre_processor(object):
	def __init__(self):
		self.p_stemmer = PorterStemmer()
		self.stopwords = set()
		self.prog_list = [re.compile('[a-zA-Z]+'), \
			re.compile(r'.+(\w)\1{2,}.?'), \
			re.compile(r'^(\w)\1{1,}.?'), \
			re.compile(r'.(\w)\1{1,}$'), \
			re.compile(r'.+oo$'), \
			re.compile(r'.+hh$'), \
			re.compile(r'.+mm$')]
		self.action_list = [None, \
			not None, \
			not None, \
			not None, \
			not None, \
			not None, \
			not None]

		print ("Building stopword dict...")
		for line in open("./stopwords/df.txt"):
			flist = line.strip().split("\t")
			self.stopwords.add(flist[1])
			if int(flist[0]) < 50000:
				break
		for line in open("./stopwords/names.txt"):
			self.stopwords.add(line.strip())
		print ("Done")

	def filtered(self, w):
		w = w.strip()
		if w in self.stopwords:
			return True
		for i in range(0, len(self.prog_list)):
			if self.prog_list[i].match(w) == None and self.action_list[i] == None:
				return True
			if self.prog_list[i].match(w) != None and self.action_list[i] != None:
				return True
		return False

	def preprocessing(self, line):
		tokens = line.strip().lower().split(" ")
		en_stop = get_stop_words('en')
		stopped_tokens = [i for i in tokens if not i in en_stop and len(i) > 2]
		final_tokends = [i for i in stopped_tokens if not self.filtered(self.p_stemmer.stem(i))]
		if len(final_tokends) == 0:
			final_tokends = tokens
		return [self.p_stemmer.stem(i) for i in final_tokends] 

if __name__ == '__main__':
	print('Loading pre-trained GloVe model') 
	g = Glove.load('glove.model')
	print('Loading finished') 

	en_file = sys.argv[1] + ".en"
	vi_file = sys.argv[1] + ".vi"
	en_raw = []; vi_raw = []
	pp = pre_processor()
	for line in open(en_file):
		line = line.decode("utf8", "ignore")
		en_raw.append(line.strip())
	for line in open(vi_file):
		line = line.decode("utf8", "ignore")
		vi_raw.append(line.strip())

	scores = {}
	for i in range(0, len(en_raw)):
		en_line = pp.preprocessing(en_raw[i])
		vi_line = pp.preprocessing(vi_raw[i])
		if len(en_line) == 0 or len(vi_line) == 0:
			continue
		score, not_unk_en, not_unk_vi = g.paragraph_similarity(en_line, vi_line)
		if score > -1:
			#pairs = "[" + " ".join(en_line) + " <-> " + " ".join(vi_line) + "]\t" + en_raw[i] + "\t" + vi_raw[i]
			pairs = en_raw[i] + "\t" + vi_raw[i]
			print(str(score) + "\t" + pairs.encode("utf8", "ignore"))
	
