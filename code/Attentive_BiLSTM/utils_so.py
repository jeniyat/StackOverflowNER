from __future__ import print_function
import os
import re
import numpy as np

from collections import Counter
import json


import torch.nn as nn
from torch.nn import init

from config_so import parameters

#so far best: 9911: f1= 48.83: epoch 24
# seed = parameters["seed"]
np.random.seed(parameters["seed"])





def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if add_label:
        input.append(data['tags'])
    return input


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    # nn.init.uniform(input_embedding, -bias, bias)
    nn.init.uniform_(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    #nn.init.uniform(input_linear.weight, -bias, bias)
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        # nn.init.uniform(weight, -bias, bias)
        nn.init.uniform_(weight, -bias, bias)

        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        # nn.init.uniform(weight, -bias, bias)
        nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            # nn.init.uniform(weight, -bias, bias)
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            # nn.init.uniform(weight, -bias, bias)
            nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1



def init_gru(input_gru):
    """
    Initialize lstm
    """
    for ind in range(0, input_gru.num_layers):
        weight = eval('input_gru.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        # nn.init.uniform(weight, -bias, bias)
        nn.init.uniform_(weight, -bias, bias)

        weight = eval('input_gru.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        # nn.init.uniform(weight, -bias, bias)
        nn.init.uniform_(weight, -bias, bias)

    if input_gru.bidirectional:
        for ind in range(0, input_gru.num_layers):
            weight = eval('input_gru.weight_ih_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            # nn.init.uniform(weight, -bias, bias)
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('input_gru.weight_hh_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            # nn.init.uniform(weight, -bias, bias)
            nn.init.uniform_(weight, -bias, bias)

    if input_gru.bias:
        for ind in range(0, input_gru.num_layers):
            weight = eval('input_gru.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_gru.hidden_size: 2 * input_gru.hidden_size] = 1
            weight = eval('input_gru.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_gru.hidden_size: 2 * input_gru.hidden_size] = 1
        if input_gru.bidirectional:
            for ind in range(0, input_gru.num_layers):
                weight = eval('input_gru.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_gru.hidden_size: 2 * input_gru.hidden_size] = 1
                weight = eval('input_gru.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_gru.hidden_size: 2 * input_gru.hidden_size] = 1


                
#------------------------------------------------------------------------------------------------------------------
#  added by JT to merge low freq labels
#------------------------------------------------------------------------------------------------------------------
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






class Sort_Entity_by_Count:
    """docstring for Sort_Entity_by_Count"""
    def __init__(self, train_file,output_file):
        l = self.Read_File(train_file)
        #
        self.list_of_train_sentence_words=l[0]
        self.list_of_train_sentence_labels=l[1]
        self.train_ques_count=l[2]
        self.train_answer_count=l[3]

        train_label_counter = Counter(x for xs in self.list_of_train_sentence_labels for x in xs)
        train_result=self.get_label_counter(train_label_counter)


        list_keys= [x[0] for x in train_result["label_phrase_counter"].most_common()]
        with open(output_file, 'w') as outfile:
            json.dump(list_keys, outfile)



    def get_label_counter(self, label_counter):
        label_phrase_counter=Counter()
        label_word_counter=Counter()

        word_count=0
        entities_count=0

        for c in label_counter:
            split_c=c.split("-",1)
            type_c=split_c[0]
            if type_c=="O":
                word_count+=label_counter[c]
                continue
            entity_name=split_c[1]
            #print(entity_name, split_c, type_c)
            if type_c=="B":
                label_phrase_counter[entity_name]+=label_counter[c]
                label_word_counter[entity_name]+=label_counter[c]
                word_count+=label_counter[c]
                entities_count+=label_counter[c]
            elif type_c=="I":
                label_word_counter[entity_name]+=label_counter[c]

        result={}
        result["label_phrase_counter"]=label_phrase_counter
        result["word_count"]=word_count
        result["entity_count"]=entities_count
        result["label_word_counter"]=label_word_counter
        return result


    def Read_File(self, ip_file):
        list_of_sentence_words_in_file=[]
        list_of_sentence_labels_in_file=[]
        current_sent_words=[]
        current_sent_labels=[]
        count_question=0
        count_answer=0

        for line in open(ip_file):
            #print(line)
            if line.startswith("Question_ID"):
                #print(line)
                count_question+=1
            if line.startswith("Answer_to_Question_ID"):
                count_answer+=1
            if line.strip()=="":
                if len(current_sent_words)>0:
                    output_line = " ".join(current_sent_words)
                    #print(output_line)
                    if "code omitted for annotation" in output_line and "CODE_BLOCK :" in output_line:
                        current_sent_words=[]
                        current_sent_labels=[]
                        continue
                    elif "omitted for annotation" in output_line and "OP_BLOCK :" in output_line:
                        current_sent_words=[]
                        current_sent_labels=[]
                        continue
                    elif "Question_URL :" in output_line:
                        current_sent_words=[]
                        current_sent_labels=[]
                        continue
                    elif "Question_ID :" in output_line:
                        current_sent_words=[]
                        current_sent_labels=[]
                        continue
                    else:
                        list_of_sentence_words_in_file.append(current_sent_words)
                        list_of_sentence_labels_in_file.append(current_sent_labels)
                        
                        current_sent_words=[]
                        current_sent_labels=[]
                    
                    
            else:
                #print(line)
                line_values=line.strip().split()
                gold_word=line_values[0]
                gold_label=line_values[1]
                raw_word=line_values[2]
                raw_label=line_values[3]
                current_sent_words.append(gold_word)
                current_sent_labels.append(gold_label)
                
        
        return [list_of_sentence_words_in_file, list_of_sentence_labels_in_file, count_question, count_answer]