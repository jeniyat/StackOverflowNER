from utils_preprocess import *
from utils_preprocess.format_markdown import *
from utils_preprocess.anntoconll import *

import glob

from utils_ctc import *
from utils_ctc.prediction_ctc import *
import argparse

import subprocess

import shutil

import softner_segmenter_preditct_from_file
import softner_ner_predict_from_file

def read_file(input_file, output_folder):

	info_extractor = Stackoverflow_Info_Extract(output_folder)

	post_id = 0

	for line in open(input_file):
		if line.strip()=="":
			continue
		post_id+=1

		# if "--INLINE_CODE_BEGIN---" in line:
		# 	print(post_id)
		
		info_extractor.tokenize_and_annotae_post_body(line,str(post_id).zfill(6))


def merge_all_conll_files(conlll_folder, output_file):
	fout = open(output_file,'w')

	list_of_text_files = [f for f in os.listdir(conlll_folder) if f.endswith('.txt')]
	# print(list_of_text_files)

	for file_name in sorted(list_of_text_files):
		file_path = conlll_folder+"/"+file_name
		for line in open(file_path):
			line=line.strip()

			
			if line=="":
				fout.write("\n")
				continue

			line_values = line.strip().split()
			# print(line_values, len(line_values))
			if len(line_values)==2:
				# print(line)
				fout.write(line)
				fout.write("\n")
		fout.flush()
		fout.write("\n")

	# print(output_file)

	fout.close()


def create_segmenter_input(conll_format_file, segmenter_input_file, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features):
	


	fout=open(segmenter_input_file,'w')

	for line in open(conll_format_file):
		if line.strip()=="":
			fout.write("\n")
			continue

		# print("--------------",line)
		
		line_values = line.strip().split()
		if len(line_values)!=2:
			print(line)
			continue
		else:
			word, md = line.split()

		ctc = prediction_on_token_input(word, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features)
		# print(word, md, ctc)

		if md!="O":
			md = "Name"
		opline = word+"\t"+"O"+"\t"+"CTC_PRED:"+str(ctc)+"\t"+"md_label:"+md+"\n"
		# print(opline)
		fout.write(opline)

	fout.close()
		

def create_ner_input(segmenter_output_file, ner_input_file, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features):
	


	fout=open(ner_input_file,'w')

	for line in open(segmenter_output_file):
		if line.strip()=="":
			fout.write("\n")
			continue

		# print("--------------",line)
		
		line_values = line.strip().split()
		if len(line_values)!=2:
			print(line)
			continue
		else:
			word, seg = line.split()

		ctc = prediction_on_token_input(word, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features)
		# print(word, md, ctc)

		if seg!="O":
			seg = "Name"

		opline = word+"\t"+"O"+"\t"+"CTC_PRED:"+str(ctc)+"\t"+"pred_seg_label:"+seg+"\n"
		# print(opline)
		fout.write(opline)

	fout.close()
		

def parse_args():
    parser = argparse.ArgumentParser()


    # Required parameters
    parser.add_argument(
        "--input_file_with_so_body",
        default='xml_filted_body.txt',
        type=str,
    )

    args = parser.parse_args()

    


    return args


def Extract_NER(input_file):


	train_file=parameters_ctc['train_file']
	test_file=parameters_ctc['test_file']
	
	ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features= train_ctc_model(train_file, test_file)


	base_temp_dir = "temp_files/"
	standoff_folder = "temp_files/standoff_files/"
	conlll_folder = "temp_files/conll_files/"
	conll_file = "temp_files/conll_format_txt.txt"

	segmenter_input_file = "temp_files/segmenter_ip.txt"

	segmenter_output_file = "temp_files/segemeter_preds.txt"

	ner_input_file = "temp_files/ner_ip.txt"

	ner_output_file = "ner_preds.txt"

	if not os.path.exists(base_temp_dir): os.makedirs(base_temp_dir)
	if not os.path.exists(standoff_folder): os.makedirs(standoff_folder)
	if not os.path.exists(conlll_folder): os.makedirs(conlll_folder)

	
	
	

	read_file(input_file, standoff_folder)
	
	convert_standoff_to_conll(standoff_folder, conlll_folder)
	merge_all_conll_files(conlll_folder, conll_file)

	create_segmenter_input(conll_file, segmenter_input_file, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features)

	softner_segmenter_preditct_from_file.predict_segments(segmenter_input_file, segmenter_output_file)

	create_ner_input(segmenter_output_file, ner_input_file, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features)

	softner_ner_predict_from_file.predict_entities(ner_input_file,ner_output_file)


	

	shutil.rmtree(base_temp_dir, ignore_errors=True)


if __name__ == '__main__':

	args = parse_args()
	input_file = args.input_file_with_so_body

	input_file = "xml_filted_body.txt"


	Extract_NER(input_file)

	



