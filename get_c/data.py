#coding=utf8
import pprint
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
		'''
		for line in open("./stopwords/df.txt"):
			flist = line.strip().split("\t")
			self.stopwords.add(flist[1])
			if int(flist[0]) < 50000:
				break
		'''
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
		return [i for i in stopped_tokens if not self.filtered(self.p_stemmer.stem(i))]

if __name__ == '__main__':
	pp = pre_processor()
	fpout = open("Disc_train.dat", "w")
	for line in open("../data/filter/bag_of_words"):
		flist = line.strip().split("\t")
		if len(flist) == 2:
			fpout.write(" ".join(pp.preprocessing(flist[1].decode("utf8", "ignore"))) + "\n")
	fpout.close()
