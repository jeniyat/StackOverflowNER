import sys
import codecs
import os


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
	def __init__(self,  annotattion_folder):
		
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
				
				temp_soup.pre.string="CODE_BLOCK: id_"+str(self.code_file_number)+" (code omitted for annotation)\n"
				
				



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
							
							inline_code.string="CODE_BLOCK: id_"+str(self.code_file_number)+" (code omitted for annotation)\n"
							
							

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

	def tokenize_and_annotae_post_body(self, xml_filtered_string, post_id):

		text_file_name=self.annotattion_folder+post_id+".txt"
		ann_file_name=self.annotattion_folder+post_id+".ann"

		f_out_txt=open(text_file_name,'w')
		f_out_ann=open(ann_file_name,'w')

		#print "-----------------------"


		tokenized_body=tokenizer.tokenize(xml_filtered_string)

		#tokenized_body_extra_stop_removed = [sentence.replace("..",".") for sentence in tokenized_body if "--INLINE_CODE_BEGIN---" in sentence]
		tokenized_body_extra_qmark_and_stop_removed = []
		for sentence in tokenized_body:
			if "--INLINE_CODE_BEGIN---" in sentence:
				sentence=sentence.replace("..",".")
				sentence = sentence.replace("<?-?>","?")
			tokenized_body_extra_qmark_and_stop_removed.append(sentence)

		#tokenized_body_extra_q_mark_removed = [sentence.replace("<?-?>","?") for sentence in tokenized_body_extra_stop_removed]
		tokenized_body_new_line_removed=[re.sub(r"\n+", "\n", sentence) for sentence in tokenized_body_extra_qmark_and_stop_removed]
		#print tokenized_body
		#print tokenized_body_new_line_removed
		tokenized_body_str="\n".join(tokenized_body_new_line_removed)
		#print tokenized_body_str

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
		intro_text=""

		f_out_txt.write(intro_text)
		op_string=tokenized_body_str.replace("--INLINE_CODE_BEGIN---","").replace("--INLINE_CODE_END---","").replace("--KEYBOARD_IP_BEGIN---","").replace("--KEYBOARD_IP_END---","")
		f_out_txt.write(op_string)
		f_out_txt.write("\n")
		f_out_txt.close()

		#-----------------creating automated annotations---------------------------
		tag_counter=1
		init_offset=len(intro_text)

		len_keyboard_tag_str=len("--KEYBOARD_IP_BEGIN---"+"--KEYBOARD_IP_END---")
		len_inline_code_tag_str= len("--INLINE_CODE_BEGIN---" +"--INLINE_CODE_END---")





		#-----------------annotation for inline codes------------------------------
		
		number_of_inline_codes=len(inline_code_begining_positions)
		
		inline_code_tag_offset=len("--INLINE_CODE_BEGIN---")+len("--INLINE_CODE_END---")

		
		for postion in range(0,number_of_inline_codes):
			begining_pos=inline_code_begining_positions[postion]
			ending_pos=inline_code_ending_positions[postion]
			code_string= tokenized_body_str[begining_pos:ending_pos].replace("--INLINE_CODE_BEGIN---","").replace("--INLINE_CODE_END---","").replace('\r', '').replace('\n', '')
			#print(code_string)
			tag_text="T"+str(tag_counter)
			annotation_text="Code_Block"

			adjusted_begining_pos = begining_pos 
			adjusted_ending_pos = ending_pos
			

			#---------adjust annoatiotion location for keyboard ips------------------------
			for keyboard_ip_pos in keyboard_ip_begining_positions:
				if begining_pos > keyboard_ip_pos:
					adjusted_begining_pos= adjusted_begining_pos - len_keyboard_tag_str
					adjusted_ending_pos= adjusted_ending_pos - len_keyboard_tag_str
				
			begin_loc=adjusted_begining_pos+init_offset
			ending_loc=adjusted_ending_pos+init_offset


			begin_loc=adjusted_begining_pos+init_offset-(postion*inline_code_tag_offset)
			ending_loc=adjusted_ending_pos+init_offset-((postion+1)*inline_code_tag_offset)
			
			
			opline=tag_text+"\t"+annotation_text+" "+str(begin_loc)+" "+str(ending_loc)+"\t"+code_string+"\n"
			#print("************")
			#print(opline)
			f_out_ann.write(opline)
			tag_counter+=1



		#-----------------annotation for output block------------------------------

		number_of_op_blocks=len(op_block_begining_positions)

		for postion in range(0,number_of_op_blocks):
			begining_pos=op_block_begining_positions[postion]
			ending_pos=op_block_ending_positions[postion]
			op_string= tokenized_body_str[begining_pos:ending_pos]
			#print(code_string)
			tag_text="T"+str(tag_counter)
			annotation_text="Output_Block"

			adjusted_begining_pos = begining_pos 
			adjusted_ending_pos = ending_pos
			#---------adjust annoatiotion location for inline code---------------
			for inline_code_pos in inline_code_begining_positions:
				#print "DBUG---inline code inline_code_begining_position :",inline_code_pos, type(inline_code_pos), begining_pos, type(begining_pos)
				if begining_pos > inline_code_pos:
					adjusted_begining_pos= adjusted_begining_pos - len_inline_code_tag_str
					adjusted_ending_pos= adjusted_ending_pos - len_inline_code_tag_str
					#print "DBUG---if of inline code adjust :", adjusted_begining_pos, adjusted_ending_pos

			#---------adjust annoatiotion location for keyboard ips------------------------
			for keyboard_ip_pos in keyboard_ip_begining_positions:
				if begining_pos > keyboard_ip_pos:
					adjusted_begining_pos= adjusted_begining_pos - len_keyboard_tag_str
					adjusted_ending_pos= adjusted_ending_pos - len_keyboard_tag_str
				
			begin_loc=adjusted_begining_pos+init_offset
			ending_loc=adjusted_ending_pos+init_offset
			
			
			opline=tag_text+"\t"+annotation_text+" "+str(begin_loc)+" "+str(ending_loc)+"\t"+op_string+"\n"
			#print("************")
			#print(opline)
			f_out_ann.write(opline)
			tag_counter+=1

		#-----------------annotation for keyboard input------------------------------
		number_of_keyboard_ips=len(keyboard_ip_begining_positions)
		
		keyboard_tag_offset=len("--KEYBOARD_IP_BEGIN---")+len("--KEYBOARD_IP_END---")

		
		for postion in range(0,number_of_keyboard_ips):
			begining_pos=keyboard_ip_begining_positions[postion]
			ending_pos=keyboard_ip_ending_positions[postion]
			keyboard_ip_string= tokenized_body_str[begining_pos:ending_pos].replace("--KEYBOARD_IP_BEGIN---","").replace("--KEYBOARD_IP_END---","")
			#print(code_string)
			tag_text="T"+str(tag_counter)
			annotation_text="Keyboard_IP"

			adjusted_begining_pos = begining_pos 
			adjusted_ending_pos = ending_pos

			#---------adjust annoatiotion location for inline code---------------
			for inline_code_pos in inline_code_begining_positions:
				#print "DBUG---inline code inline_code_begining_position :",inline_code_pos, type(inline_code_pos), begining_pos, type(begining_pos)
				if begining_pos > inline_code_pos:
					adjusted_begining_pos= adjusted_begining_pos - len_inline_code_tag_str
					adjusted_ending_pos= adjusted_ending_pos - len_inline_code_tag_str
					#print "DBUG---if of inline code adjust :", adjusted_begining_pos, adjusted_ending_pos

		
			

			begin_loc=adjusted_begining_pos+init_offset-(postion*keyboard_tag_offset)
			ending_loc = adjusted_ending_pos+init_offset-((postion+1)*keyboard_tag_offset)
			
			
			opline=tag_text+"\t"+annotation_text+" "+str(begin_loc)+" "+str(ending_loc)+"\t"+keyboard_ip_string+"\n"
			#print("************")
			#print(opline)
			f_out_ann.write(opline)
			tag_counter+=1

		#-----------------annotation for code block------------------------------

		number_of_code_blocks=len(code_block_begining_positions)
		#print "DEBUG--codeblock"
		#print "DEBUG--question id : ",post_id
		#print "DEBUG---inputs:",tokenized_body_str
		#print "DEBUG---input len:",len(tokenized_body_str)
		#print "DEBUG---itro text len:",len(intro_text)

		#print "DBUG---code_block_begining_positions:",code_block_begining_positions
		#print "DBUG---code_block_ending_positions:",code_block_ending_positions
		#print "DBUG---init offset :",init_offset

		for postion in range(0,number_of_code_blocks):
			begining_pos=code_block_begining_positions[postion]
			ending_pos=code_block_ending_positions[postion]
			code_string= tokenized_body_str[begining_pos:ending_pos]
			#print(code_string)
			tag_text="T"+str(tag_counter)
			annotation_text="Code_Block"

			adjusted_begining_pos = begining_pos 
			adjusted_ending_pos = ending_pos
			#---------adjust annoatiotion location for inline code---------------
			for inline_code_pos in inline_code_begining_positions:
				#print "DBUG---inline code inline_code_begining_position :",inline_code_pos, type(inline_code_pos), begining_pos, type(begining_pos)
				if begining_pos > inline_code_pos:
					adjusted_begining_pos= adjusted_begining_pos - len_inline_code_tag_str
					adjusted_ending_pos= adjusted_ending_pos - len_inline_code_tag_str
					#print "DBUG---if of inline code adjust :", adjusted_begining_pos, adjusted_ending_pos

			#---------adjust annoatiotion location for keyboard ips------------------------
			for keyboard_ip_pos in keyboard_ip_begining_positions:
				if begining_pos > keyboard_ip_pos:
					adjusted_begining_pos= adjusted_begining_pos - len_keyboard_tag_str
					adjusted_ending_pos= adjusted_ending_pos - len_keyboard_tag_str
				
			begin_loc=adjusted_begining_pos+init_offset
			ending_loc=adjusted_ending_pos+init_offset
			
			#print "DEBUG--begining postion+init offset : ",begin_loc, begining_pos+init_offset
			#print "DEBUG--ending postion+init offset : ",ending_loc, begining_pos+init_offset 
			opline=tag_text+"\t"+annotation_text+" "+str(begin_loc)+" "+str(ending_loc)+"\t"+code_string+"\n"
			#print("************")
			#print "DEBUG--code block ann : ",opline
			f_out_ann.write(opline)
			tag_counter+=1
		#print "------------DEBUG-----",post_id
		f_out_ann.close()

		





		
		

		
		