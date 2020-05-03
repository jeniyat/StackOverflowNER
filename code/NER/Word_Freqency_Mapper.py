from gaussian_binner import GaussianBinner
import numpy as np
from collections import Counter

class Word_Freqency_Mapper:
	"""docstring for ClassName"""
	def __init__(self,bins=100, w=5.0):
		self.Train_Word_Counter=Counter()
		self.set_of_freq=set()
		self.binner = GaussianBinner(bins=bins, w=w)

		self.all_word_w_freq_vector={}
		self.Test_Set_Words=set()

	def Find_Freq_Vector_for_words(self):
		
		ip_array=np.array([0])

		for word in self.Train_Word_Counter:
			word_freq=np.array([self.Train_Word_Counter[word]])
			temp_ip_array = np.vstack((ip_array,word_freq))
			freq_vector= self.binner.transform(temp_ip_array,1)[1]
			self.all_word_w_freq_vector[word]=freq_vector
			# print(freq_vector.shape)
			# print(word)
			# print(freq_vector[1])
		# print(self.Test_Set_Words - (set()))
		# print(len(self.Train_Word_Counter.keys()))
		for word in self.Test_Set_Words:
			if word in self.Train_Word_Counter:
				# continue
				# word_freq1= self.all_word_w_freq_vector[word]
				word_freq=np.array([self.Train_Word_Counter[word]])
				# print(np.subtract(word_freq1, word_freq1))
				# print("\n\n\n")
				continue
			else:
				word_freq=np.array([0])
				# print(word, word_freq)

			temp_ip_array = np.vstack((ip_array,word_freq))
			freq_vector= self.binner.transform(temp_ip_array,1)[1]
			# print(word, freq_vector)
			self.all_word_w_freq_vector[word]=freq_vector
		# print(self.all_word_w_freq_vector)

	def Read_File(self, ip_file):
		list_of_sentence_words_in_file=[]
		list_of_sentence_labels_in_file=[]
		list_of_markdown_markdowns_in_file=[]
		current_sent_words=[]
		current_sent_labels=[]
		current_sent_markdowns=[]

		for line in open(ip_file):
			#print(line)
			if line.strip()=="":
				if len(current_sent_words)>0:
					output_line = " ".join(current_sent_words)
					#print(output_line)
					if "code omitted for annotation" in output_line and "CODE_BLOCK :" in output_line:
						current_sent_words=[]
						current_sent_labels=[]
						current_sent_markdowns=[]
						continue
					elif "omitted for annotation" in output_line and "OP_BLOCK :" in output_line:
						current_sent_words=[]
						current_sent_labels=[]
						current_sent_markdowns=[]
						continue
					elif "Question_URL :" in output_line:
						current_sent_words=[]
						current_sent_labels=[]
						current_sent_markdowns=[]
						continue
					elif "Question_ID :" in output_line:
						current_sent_words=[]
						current_sent_labels=[]
						current_sent_markdowns=[]
						continue
					else:
						list_of_sentence_words_in_file.append(current_sent_words)
						list_of_sentence_labels_in_file.append(current_sent_labels)
						list_of_markdown_markdowns_in_file.append(current_sent_markdowns)
						current_sent_words=[]
						current_sent_labels=[]
						current_sent_markdowns=[]
					
					
			else:
				line_values=line.strip().split()
				gold_word=line_values[0]
				gold_label=line_values[1]
				raw_word=line_values[2]
				raw_label=line_values[3]

				word=gold_word
				label=gold_label

				current_sent_words.append(word)
				current_sent_markdowns.append(raw_label)
				current_sent_labels.append(label)
				
		return (list_of_sentence_words_in_file, list_of_sentence_labels_in_file, list_of_markdown_markdowns_in_file)

	def Read_Test_Data(self, input_test_file):
		(list_of_sentence_words_in_file, list_of_sentence_labels_in_file, list_of_markdown_markdowns_in_file) = self.Read_File(input_test_file)
		

		for list_of_words in list_of_sentence_words_in_file:
			for word in list_of_words:
				self.Test_Set_Words.add(word)

	def Read_Dev_Data(self, input_dev_file):
		(list_of_sentence_words_in_file, list_of_sentence_labels_in_file, list_of_markdown_markdowns_in_file) = self.Read_File(input_dev_file)
		

		for list_of_words in list_of_sentence_words_in_file:
			for word in list_of_words:
				self.Test_Set_Words.add(word)


	def Find_Train_Data_Freq(self, input_train_file):

		(list_of_sentence_words_in_file, list_of_sentence_labels_in_file, list_of_markdown_markdowns_in_file) = self.Read_File(input_train_file)
		for list_of_words in list_of_sentence_words_in_file:
			for word in list_of_words:
				self.Train_Word_Counter[word]+=1

		for word in self.Train_Word_Counter:
			self.set_of_freq.add(self.Train_Word_Counter[word])

		# print(min(self.set_of_freq))


	def Find_Gaussian_Bining_For_Training_Data_Freq(self):
		freq_array=np.array([0])

		for word in self.Train_Word_Counter:
			word_freq =np.array([self.Train_Word_Counter[word]])
			freq_array = np.vstack((freq_array,word_freq ))


		self.binner.fit(freq_array, 1)

	def Write_Freq_To_File(self,output_file):
		fout= open(output_file,'w')
		for word in self.all_word_w_freq_vector:
			word_freq=self.all_word_w_freq_vector[word].tolist()
			word_freq_str=[str(x) for x in word_freq]
			# print(word_freq)
			# word_freq_str = np.array2string(word_freq, formatter={'str':lambda x: float(word_freq)})
			# print(word_freq_str)
			# print("\n\n\n")
			opline=word+" "+" ".join(word_freq_str)+"\n"
			# print(opline)
			fout.write(opline)
		fout.close()





if __name__ == '__main__':
	input_train_file="../../StackOverflow_Input_Data/train_gold_raw_merged.txt"
	input_dev_file="../../StackOverflow_Input_Data/dev_gold_raw_merged.txt"
	input_test_file="../../StackOverflow_Input_Data/test_gold_raw_merged.txt"
	output_file="Freq_Vector.txt"

	freq_mapper = Word_Freqency_Mapper()
	freq_mapper.Find_Train_Data_Freq(input_train_file)
	freq_mapper.Read_Dev_Data(input_dev_file)
	freq_mapper.Read_Test_Data(input_test_file)
	freq_mapper.Find_Gaussian_Bining_For_Training_Data_Freq()
	freq_mapper.Find_Freq_Vector_for_words()
	freq_mapper.Write_Freq_To_File(output_file)









