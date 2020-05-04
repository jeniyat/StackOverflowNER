import json
import sys


def Merge_Label(inputFile):
    merging_dict={}
    merging_dict["Library_Function"]="Function"
    merging_dict["Function_Name"]="Function"

    merging_dict["Class_Name"]="Class"
    merging_dict["Library_Class"]="Class"

    merging_dict["Library_Variable"]="Variable"
    merging_dict["Variable_Name"]="Variable"

    merging_dict["Website"]="Website"
    merging_dict["Organization"]="Website"

    modified_file=inputFile[:-4]+"_merged_labels.txt"
    Fout=open(modified_file,"w")
    line_count=0
    for line in open(inputFile):
        line_count+=1
        # print("line: in Merge_Label: utils_so:  ",line)
        # print(inputFile,":", line_count)
        line_values=line.strip().split()
        if len(line_values)<2:
            opline=line
            Fout.write(opline)
            continue
            

        gold_word=line_values[0]
        gold_label=line_values[1]
        raw_word=line_values[2]
        raw_label=line_values[3]
        #print(line_values)
        if gold_word!=raw_word:
            print("wrong mapping: ", line)

        word=gold_word
        label=gold_label

        if label=="O":
            opline=line
            Fout.write(opline)
            continue
        # print(label)

        label_split=label.split("-",1)

        label_prefix=label_split[0]
        label_name=label_split[1]
        #print(label_name)
        
        if label_name in merging_dict:
            label_name=merging_dict[label_name]
            #print(label_name)

        new_label=label_prefix+"-"+label_name
        #opline=word+" "+new_label+"\n"
        opline=word+" "+new_label+" "+raw_word+" "+raw_label+"\n"
        Fout.write(opline)


    Fout.close()
    return modified_file




    return modified_file




def loader_so_text(path, merge_tag=True, replace_low_freq_tags=True):
	if merge_tag:
		path=Merge_Label(path)

	set_of_selected_tags = []

	if replace_low_freq_tags:
		sorted_entity_list = ["Class","Class_Name", "Library_Class", "Application", "Library_Variable", "Variable_Name", "Variable", "User_Interface_Element", "Code_Block", "Library_Function","Function_Name", "Function", "Language", "Library", "Data_Structure", "Data_Type", "File_Type", "File_Name", "Version", "HTML_XML_Tag", "Device", "Operating_System", "User_Name", "Website", "Output_Block", "Error_Name", "Algorithm", "Organization", "Keyboard_IP", "Licence", "Organization"]
		set_of_selected_tags.extend(sorted_entity_list[0:-6])
		if 'Algorithm' not in set_of_selected_tags: set_of_selected_tags.append('Algorithm')


	sentence = [] #list of words in the current sentence in formate each word list looks like [word, markdow tag name, mark down tag, NER tag]
	sentences = [] #list of sentences



	count_question=0
	count_answer=0


	max_len=0

	for line in open(path):
		if line.startswith("Question_ID"):
			count_question+=1

		if line.startswith("Answer_to_Question_ID"):
			count_answer+=1

		if line.strip()=="":
			if len(sentence) > 0:
				output_line = " ".join(w[0] for w in sentence)

				if "code omitted for annotation" in output_line and "CODE_BLOCK :" in output_line:
					sentence = []
					continue

				elif "omitted for annotation" in output_line and "OP_BLOCK :" in output_line:
					sentence = []
					continue

				elif "Question_URL :" in output_line:
					sentence = []
					continue

				elif "Question_ID :" in output_line:
					sentence=[]
					continue

				else:
					sentences.append(sentence)
					if len(sentence)>max_len:
						max_len=len(sentence)

					sentence=[]

		else:

			line_values=line.strip().split()

			gold_word=line_values[0]
			gold_label=line_values[1]
			raw_word=line_values[2]
			raw_label=line_values[3]

			gold_word=" ".join(gold_word.split('-----'))

			gold_label_name= gold_label.replace("B-","").replace("I-","")

			if gold_label_name not in set_of_selected_tags:
				gold_label="O"

			word_info=[gold_word, raw_label, gold_label]
			sentence.append(word_info)
	print("------------------------------------------------------------")
	print("Number of questions in ", path, " : ", count_question)
	print("Number of answers in ", path, " : ", count_answer)
	print("Number of sentences in ", path, " : ", len(sentences))
	print("Max len sentences has", max_len, "words")
	print("------------------------------------------------------------")
	return sentences

        
    

if __name__ == '__main__':

	path_to_file = "../../resources/annotated_ner_data/StackOverflow/train.txt"
	merge_tag= True
	replace_low_freq_tags= True

	all_sentneces = loader_so_text(path_to_file, merge_tag, replace_low_freq_tags)




