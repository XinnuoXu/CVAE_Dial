#coding=utf8

import sys
import json
import random

if __name__ == '__main__':
	data_list = []
	for line in open(sys.argv[1] + ".dial.jsons.txt"):
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
			disc_pos = " ".join(c_list) + "\t" + response + "\n"
			gen_en = " ".join(c_list) + "\n"
			gen_vi = response + "\n"
			data_list.append([disc_pos, gen_en, gen_vi, ""])
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
					disc_pos = " ".join(c_list) + "\t" + response + "\n"
					gen_en = " ".join(c_list) + "\n"
					gen_vi = response + "\n"

					cset = set(context)
					rset = set(response)
					sampled_response = random.sample(list(fset - cset - rset), 1)[0]
					disc_neg = " ".join(c_list) + "\t" + sampled_response + "\n"
					data_list.append([disc_pos, gen_en, gen_vi, disc_neg])
				except:
					continue
	
	suf = random.shuffle(data_list)

	fp_gen_en = open(sys.argv[1] + ".en", "w")
	fp_gen_vi = open(sys.argv[1] + ".vi", "w")
	fp_disc_pos = open(sys.argv[1] + ".pos", "w")
	fp_disc_neg = open(sys.argv[1] + ".neg", "w")

	count = 0
	for item in data_list:
		if count < 5000:
			try:
				fp_disc_pos.write(item[0])
				fp_gen_en.write(item[1])
				fp_gen_vi.write(item[2])
				fp_disc_neg.write(item[3])
			except:
				continue
		else:
			break
		count += 1

	fp_gen_en.close()
	fp_gen_vi.close()
	fp_disc_pos.close()
	fp_disc_neg.close()
