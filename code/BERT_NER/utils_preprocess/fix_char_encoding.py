import ftfy
import stokenizer
import ark_twokenize

class Fix_Char_Code:
	"""docstring for Fix_Char_Code"""
	def __init__(self):
		pass

	def Get_List_of_Labels(self, tokenized_word_list_len, main_label):
		if main_label=="O":
			new_label="O"
		elif main_label[0]=="B":
			new_label=main_label.replace("B-","I-")
		else:
			new_label= main_label

		new_label_list=[main_label]
		for i in range(tokenized_word_list_len-1):
			new_label_list.append(new_label)
		# print(tokenized_word_list_len, main_label, new_label_list)
		return new_label_list

	def Fix_Word_Label(self, word, gold_label, raw_label):
		if "&zwnj" in word or "&nbsp" in word or "&amp" in word:
			return ([word], [gold_label], [raw_label], False)

		fixed_word = ftfy.fix_text(word)
		#the following line is found from error analysis over the fixed encoding by finding  && in the text file

		fixed_word=fixed_word.replace("´","'").replace("ÂŁ","£").replace('Ăż','ÿ').replace('Âż','¿').replace('ÂŹ','¬').replace('รก','á').replace("â","†").replace("`ĚN","`̀N")
		modified=True
		if fixed_word==word:
			modified = False
			return ([fixed_word], [gold_label], [raw_label], modified)
		try:
			fixed_word_tokenized= stokenizer.tokenize(fixed_word)

		except stokenizer.TimedOutExc as e:
			try:
				fixed_word_tokenized= ark_twokenize.tokenizeRawTweetText(fixed_word)
			except Exception as e:
				print(e)
		if len(fixed_word_tokenized)==2 and fixed_word_tokenized[0]=="'":
			return ([fixed_word], [gold_label], [raw_label],modified)
		
		# print(word, fixed_word, fixed_word_tokenized)
		new_gold_label_list = self.Get_List_of_Labels(len(fixed_word_tokenized), gold_label)
		new_raw_label_list = self.Get_List_of_Labels(len(fixed_word_tokenized), raw_label)
		return (fixed_word_tokenized, new_gold_label_list, new_raw_label_list,modified)

	def Read_File(self, ip_file):
		output_file_name = ip_file[:-4]+"_char_embed_resolved.txt"
		fout= open(output_file_name,'w')

		
		

		for line in open(ip_file):
			if line.strip()=="":
				fout.write(line)
				continue

			line_values=line.strip().split()
			gold_word=line_values[0]
			gold_label=line_values[1]
			raw_word=line_values[2]
			raw_label=line_values[3]
			(new_tokenized_word_list, new_gold_label_list, new_raw_label_list, if_modified) = self.Fix_Word_Label(gold_word, gold_label, raw_label)
			if if_modified:
				print(line.strip())
				print(new_tokenized_word_list)
				print("----")
			# print(new_tokenized_word_list, new_gold_label_list, new_raw_label_list)
			for word_iter in range(len(new_tokenized_word_list)):
				word = new_tokenized_word_list[word_iter]
				if word.strip()=="":
					continue

				gold_label = new_gold_label_list[word_iter]

				if word == "'s":
					gold_label="O"

				raw_label = new_raw_label_list[word_iter]
				op_line = word+"\t"+gold_label+"\t"+word+"\t"+raw_label+"\n"
				# print(op_line)
				fout.write(op_line)

			
		
		


		
if __name__ == '__main__':
	fcc = Fix_Char_Code()

	ip_file_name = "test_gold_raw_merged_04_05.txt"
	fcc.Read_File(ip_file_name)

	ip_file_name = "train_gold_raw_merged_04_05.txt"
	fcc.Read_File(ip_file_name)


	ip_file_name = "dev_gold_raw_merged_04_05.txt"
	fcc.Read_File(ip_file_name)



