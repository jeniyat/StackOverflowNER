import tolatex
import json

import utils_so as utils 


from config_so import parameters

def print_result(eval_result,epoch_count, sorted_entity_list_file, entity_category_code, entity_category_human_language):
	result={}
	result["header"]=["", "Precision", "Recall", "F1", "Predicted", "Correctly Predicted"]
	result["rows"]=[]
	
	over_all_result= eval_result['overall']
	by_category_result = eval_result['by_category']
	#load_sorted_entity_file
	with open(sorted_entity_list_file) as f:
		sorted_entity_list = json.load(f)


	for entity in sorted_entity_list: 
		if entity not in by_category_result: 
			continue
		l = [entity, by_category_result[entity]["P"], by_category_result[entity]["R"], by_category_result[entity]["F1"], by_category_result[entity]["Total Predicted"], by_category_result[entity]["Correctly Predicted"] ]
		result["rows"].append(l)
	
	l=["overall", over_all_result["P"],over_all_result["R"], over_all_result["F1"], over_all_result["Total Predicted"], over_all_result["Correctly Predicted"]]
	result["rows"].append(l)
	#tolatex.tolatex(result)

	# global epoch_count
	# epoch_count+=1
	op_str=[str(elem) for elem in l]
	
	# Fout_Perf_by_Epoch = open(utils.perf_per_epoch_file_all,'a')
	# Fout_Perf_by_Epoch.write("epoch_count:"+ str(epoch_count)+ "\t" +"\t".join(op_str)+"\n")
	# Fout_Perf_by_Epoch.close()


	result={}
	result["header"]=["", "Precision", "Recall", "F1"]
	result["rows"]=[]
	
	over_all_result= eval_result['overall']
	by_category_result = eval_result['by_category']
	
	count_code=0
	p_code=0.0
	r_code=0.0
	f1_code=0.0

	if not parameters['segmentation_only']:

		for entity in sorted_entity_list: 
			if entity not in by_category_result: 
				continue
			# if entity not in entity_category_code:
			# 	continue

			p_code+=float(by_category_result[entity]["P"])
			r_code+=float(by_category_result[entity]["R"])
			f1_code+=float(by_category_result[entity]["F1"])

			count_code+=1

			l = [entity, by_category_result[entity]["P"], by_category_result[entity]["R"], by_category_result[entity]["F1"]]
			result["rows"].append(l)

		# l=["Code Entity Avg", round((p_code/count_code),2), round((r_code/count_code),2), round((f1_code/count_code),2)]
		# result["rows"].append(l)

		# count_human_lang=0
		# p_human_lang=0.0
		# r_human_lang=0.0
		# f1_human_lang=0.0
		# for entity in sorted_entity_list: 
		# 	if entity not in by_category_result: 
		# 		continue
		# 	if entity not in entity_category_human_language:
		# 		continue

		# 	p_human_lang+=float(by_category_result[entity]["P"])
		# 	r_human_lang+=float(by_category_result[entity]["R"])
		# 	f1_human_lang+=float(by_category_result[entity]["F1"])

		# 	count_human_lang+=1

		# 	l = [entity, by_category_result[entity]["P"], by_category_result[entity]["R"], by_category_result[entity]["F1"]]
		# 	result["rows"].append(l)

		# l=["Human Lang Entity Avg", round((p_human_lang/count_human_lang),2), round((r_human_lang/count_human_lang),2), round((f1_human_lang/count_human_lang),2)]
		# result["rows"].append(l)

	l=["overall",over_all_result["P"],over_all_result["R"], over_all_result["F1"]]
	result["rows"].append(l)

	tolatex.tolatex(result)


