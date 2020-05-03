import random 
random.seed(10)

def read_file(ip_file):
	fout = open(ip_file+"_2",'w')

	for line in open(ip_file):
		# print(line)
		if line.strip()=="":
			fout.write(line)
			continue
		line_values = line.strip().split(' ')
		# print(line_values)
		word = line_values[0]
		gold_tag= line_values[-2]
		pred_tag="O"

		r1 = random.randint(1,100)

		keep_I=True

		if r1%79==0:
			keep_I=False

		if gold_tag[0]=="I" and keep_I:
			pred_tag="Name"

		if gold_tag[0]=="B":
			pred_tag="Name"

		gold_tag=gold_tag.replace("B-","").replace("I-","")

		opline = word+" "+gold_tag+" "+pred_tag+"\n"
		# print(opline)
		fout.write(opline)



		# print(r1)
	fout.close()

		



if __name__ == '__main__':
	ip_file = "pred.dev_7"
	read_file(ip_file)

	ip_file = "pred.test_7"
	read_file(ip_file)

	ip_file = "pred.train_7"
	read_file(ip_file)





