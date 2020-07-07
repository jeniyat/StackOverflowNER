import sys
import codecs



#from lxml import etree
#import lxml.etree.ElementTree as ET
import nltk
import os
from xmlr import xmliter
import re

from bs4 import BeautifulSoup

import csv

import unicodedata

import json

from collections import Counter

from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
abbreviation = ['u.s.a', 'fig', 'etc', 'eg', 'mr', 'mrs', 'e.g', 'no', 'vs', 'i.e']
punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)



def find_string_indices(input_string,string_to_search):
	string_indices=[]
	location=-1
	while True:
	    location = input_string.find(string_to_search, location + 1)
	    if location == -1: break
	    string_indices.append(location)
	return string_indices




class Stackoverflow_Info_Extract:
	def __init__(self,annotattion_folder):
		self.code_file_number = 1

		
		self.annotattion_folder=annotattion_folder


	


	def Extract_Text_From_XML(self,input_text):
	
		input_text_str = input_text.encode("utf-8").strip()
		extracted_text=""

		

		
		soup = BeautifulSoup(input_text_str,"lxml")
		all_tags = soup.find_all(True)
		for para in soup.body:
			text_for_current_block=str(para)
			
			#
			temp_soup = BeautifulSoup(text_for_current_block,"lxml")
			list_of_tags = [tag.name for tag in temp_soup.find_all()]
			tag_len=len(list_of_tags)

			if set(list_of_tags)==set(['html', 'body', 'pre', 'code']):
				
				code_string = temp_soup.pre.string
				
				temp_soup.pre.string="CODE_BLOCK: Q_"+str(self.code_file_number)+" (code omitted for annotation)\n"
				



			elif "code" in list_of_tags:
				all_inline_codes=temp_soup.find_all("code")
				
				for inline_code in all_inline_codes:
					inline_code_string_raw=str(inline_code)
					
					temp_code_soup = BeautifulSoup(inline_code_string_raw,"lxml")
					inline_code_string_list_of_text = temp_code_soup.findAll(text=True)
					inline_code_string_text ="".join(inline_code_string_list_of_text).strip()
					

					try:
						if "\n" in inline_code_string_text:
							code_string = inline_code_string_text
							
							inline_code.string="CODE_BLOCK: Q_"+str(self.code_file_number)+" (code omitted for annotation)\n"
							

						elif inline_code_string_text.count('.')>=1:
							inline_code.string = "--INLINE_CODE_BEGIN---"+ inline_code_string_text.replace(".","..").replace('\r', '').replace('\n', '') +"--INLINE_CODE_END---"

						elif inline_code_string_text.count('?')>=1:
							inline_code.string = "--INLINE_CODE_BEGIN---"+ inline_code_string_text.replace("?","<?-?>").replace('\r', '').replace('\n', '') +"--INLINE_CODE_END---"
						else:
							inline_code.string = "--INLINE_CODE_BEGIN---"+ inline_code_string_text.replace('\r', '').replace('\n', '') +"--INLINE_CODE_END---"
					except Exception as e:
						
						print("DEBUG----- inisde except for inline code -------- error", e)
						
						continue

					
						





			if "blockquote" in list_of_tags:
				
				op_string = temp_soup.blockquote.string
				temp_soup.blockquote.string="OP_BLOCK: (output omitted for annotation)\n"
				
				

			if "kbd" in list_of_tags:
				all_keyboard_ip=temp_soup.find_all("kbd")
				# print("DEBUG-keyboard-input", all_keyboard_ip)
				# print("DEBUG---inputxml: ",input_text_str)
				for keyboard_ip in all_keyboard_ip:
					print("DEBUG-keyboard-input", keyboard_ip.string)
					keyboard_ip.string = "--KEYBOARD_IP_BEGIN---"+ keyboard_ip.string +"--KEYBOARD_IP_END---"

					# print(keyboard_ip.string)



			list_of_texts=temp_soup.findAll(text=True)
			text="".join(list_of_texts)
			
			extracted_text+=text
			extracted_text+="\n\n"

			

			
			
		# print("DEBUG--extracted-text-for-xml: \n",extracted_text)
		
		return extracted_text

	def tokenize_and_annotae_post_body(self, xml_filtered_string,post_id):

		text_file_name=self.annotattion_folder+str(post_id)+".txt"

		f_out_txt=open(text_file_name,'w')
		

		


		tokenized_body=tokenizer.tokenize(xml_filtered_string)
		
		tokenized_body_extra_qmark_and_stop_removed = []
		for sentence in tokenized_body:
			if "--INLINE_CODE_BEGIN---" in sentence:
				sentence=sentence.replace("..",".")
				sentence = sentence.replace("<?-?>","?")
			tokenized_body_extra_qmark_and_stop_removed.append(sentence)

		
		tokenized_body_new_line_removed=[re.sub(r"\n+", "\n", sentence) for sentence in tokenized_body_extra_qmark_and_stop_removed]
		
		
		tokenized_body_str="\n".join(tokenized_body_new_line_removed)
		

		#-----------------postions for inline codes------------------------------

		inline_code_begining_positions=find_string_indices(tokenized_body_str,"--INLINE_CODE_BEGIN")
		inline_code_ending_positions_=find_string_indices(tokenized_body_str,"INLINE_CODE_END---")
		len_inline_code_end_tag=len("INLINE_CODE_END---")
		inline_code_ending_positions = [x+len_inline_code_end_tag for x in inline_code_ending_positions_]

		#-----------------postions for keyboard inputs------------------------------

		keyboard_ip_begining_positions=find_string_indices(tokenized_body_str,"--KEYBOARD_IP_BEGIN")
		keyboard_ip_ending_positions_=find_string_indices(tokenized_body_str,"KEYBOARD_IP_END---")
		len_keyboard_ip_end_tag=len("KEYBOARD_IP_END---")
		keyboard_ip_ending_positions = [x+len_keyboard_ip_end_tag for x in keyboard_ip_ending_positions_]
		re

		#-----------------postions for code blocks-----------------------------

		code_block_begining_positions=find_string_indices(tokenized_body_str,"CODE_BLOCK:")
		code_block_ending_positions_=find_string_indices(tokenized_body_str,"(code omitted for annotation)")
		len_code_block_end_tag=len("(code omitted for annotation)")
		code_block_ending_positions = [x+len_code_block_end_tag for x in code_block_ending_positions_]

		#-----------------postions for output blocks------------------------------

		op_block_begining_positions=find_string_indices(tokenized_body_str,"OP_BLOCK:")
		op_block_ending_positions_=find_string_indices(tokenized_body_str,"(output omitted for annotation)")
		len_op_block_end_tag=len("(output omitted for annotation)")
		op_block_ending_positions = [x+len_op_block_end_tag for x in op_block_ending_positions_]


		question_url_text="Question_URL: "+"https://stackoverflow.com/questions/"+str(post_id)+"/"
		intro_text="Question_ID: "+str(post_id)+"\n"+question_url_text+"\n\n"

		f_out_txt.write(intro_text)
		op_string=tokenized_body_str.replace("--INLINE_CODE_BEGIN---","").replace("--INLINE_CODE_END---","").replace("--KEYBOARD_IP_BEGIN---","").replace("--KEYBOARD_IP_END---","")
		f_out_txt.write(op_string)
		f_out_txt.write("\n")
		f_out_txt.close()

		





		
		

		
			

	def read_file(self, input_file):

		line_to_add_begining_of_row="<?xml version=\"1.0\" encoding=\"utf-8\"?>"+"\n"+"<posts>"
		line_to_add_ending_of_row="</posts>"
		#print question_input_file

		for line in open(input_file,'r'):
			# print(line)
			xml_doc=line_to_add_begining_of_row+line+line_to_add_ending_of_row
			temp_xml="temp_xml.xml"
			f_temp=open(temp_xml,'w')
			f_temp.write(xml_doc)
			f_temp.flush()
			f_temp.close()

			for d in xmliter(temp_xml, 'row'):
				post_id = d['@Id'].encode("utf-8").strip()
				post_id= str(post_id.decode("utf-8") )

				post_type_id = d['@PostTypeId']

				if post_type_id == "2":
					question_id = d['@ParentId'].encode("utf-8").strip().decode("utf-8")
					# print(question_id)
					
					print("now processing answer with : ", post_id, ", from question with id: ", question_id )
					post_id = question_id+"_"+post_id

				else:
					print("now processing question with id: ", post_id )


				body = d['@Body']
				body_xml_filtered = self.Extract_Text_From_XML(body)
				
				annotated_tokenized_text=self.tokenize_and_annotae_post_body(body_xml_filtered,post_id)


if __name__ == '__main__':
	
	output_folder="text_files/"


	ip_file = "Posts_Small.xml"
	info_extractor = Stackoverflow_Info_Extract(output_folder)
	info_extractor.read_file(ip_file)



