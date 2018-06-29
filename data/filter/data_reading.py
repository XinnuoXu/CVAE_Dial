#coding=utf8

import sys
import json
import random

if __name__ == '__main__':
	fp_gen_en = open("train.en", "w")
	fp_gen_vi = open("train.vi", "w")
	fp_disc_pos = open("train.pos", "w")
	fp_disc_neg = open("train.neg", "w")
	for line in open("train.dial.jsons.txt"):
		flist = json.loads(line.strip().lower())
		fset = set(flist)
		if len(flist) < 2:
			continue
		if len(flist) <= 4:
			context = flist[:-1]
			response = flist[-1]
			c_list = []
			for i in range(len(context)-1, -1, -1):
				c_list.insert(0, context[i])
				c_list.insert(0, "<u" + str(len(context) - i) + ">")
			c_list = c_list[1:]
			try:
				fp_disc_pos.write(" ".join(c_list) + "\t" + response + "\n")
				fp_gen_en.write(" ".join(c_list) + "\n")
				fp_gen_vi.write(response + "\n")
			except:
				continue
		else:
			for j in range(0, len(flist) - 3):
				context = flist[j: j + 3]
				response = flist[j + 3]
				c_list = []
				for i in range(len(context)-1, -1, -1):
					c_list.insert(0, context[i])
					c_list.insert(0, "<u" + str(len(context) - i) + ">")
				c_list = c_list[1:]

				try:
					fp_disc_pos.write(" ".join(c_list) + "\t" + response + "\n")
					fp_gen_en.write(" ".join(c_list) + "\n")
					fp_gen_vi.write(response + "\n")

					cset = set(context)
					rset = set(response)
					sampled_response = random.sample(list(fset - cset - rset), 1)[0]
					fp_disc_neg.write(" ".join(c_list) + "\t" + sampled_response + "\n")
				except:
					continue
	fp_gen_en.close()
	fp_gen_vi.close()
	fp_disc_pos.close()
	fp_disc_neg.close()
