from collections import Counter

vocab= Counter()

def read_file(ip_file):
	for line in open(ip_file):
		# print(line)
		if line.strip()=="":
			continue
		line_values = line.strip().split(' ')
		word = line_values[0]

		vocab[word]+=1

			
			

if __name__ == '__main__':
	ip_file = "pred.dev_7"
	read_file(ip_file)

	ip_file = "pred.test_7"
	read_file(ip_file)

	ip_file = "pred.train_7"
	read_file(ip_file)

	fout = open("vocab.tsv",'w')

	for w in vocab:
		print(w, vocab[w])
		opline=w+"\t"+str(vocab[w])+"\n"
		fout.write(opline)
	fout.close()
