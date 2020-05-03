import utils_so as utils
import loader_so as loader
from config_so import parameters
from utils_so import Sort_Entity_by_Count
import json


lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

merge_tags=parameters['merge_tags']


input_train_file=utils.Merge_Label(parameters["train"])
    
Sort_Entity_by_Count(input_train_file,parameters["sorted_entity_list_file_name"])

with open(parameters["sorted_entity_list_file_name"]) as f:
    sorted_entity_list = json.load(f)

set_of_selected_tags=[]




entity_category_code=parameters["entity_category_code"]
entity_category_human_language=parameters["entity_category_human_language"]


set_of_selected_tags.extend(sorted_entity_list[0:-6])

test_sentences = loader.load_sentences_so(parameters["test"], lower, zeros,merge_tags, set_of_selected_tags)
test_sentences = loader.load_sentences_so(parameters["dev"], lower, zeros,merge_tags, set_of_selected_tags)