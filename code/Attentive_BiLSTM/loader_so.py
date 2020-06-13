from __future__ import print_function, division
import os
import re
import codecs
import unicodedata

import model
import string
import random
import numpy as np

import utils_so as utils    #JT: utils for SO

from config_so import parameters
np.random.seed(parameters["seed"])


from utils_so import create_dico, create_mapping, zero_digits, Merge_Label
from utils_so import iob2, iob_iobes




def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in string.ascii_letters + " .,;'-"
    )


def load_sentences_so(path, lower, zeros, merge_tag,set_of_selected_tags):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    count_question=0
    count_answer=0

    if merge_tag:
        path=Merge_Label(path)
    sentences = [] #list of sentences

    sentence = [] #list of words in the current sentence in formate each word list looks like [word, markdow tag name, mark down tag, NER tag]
    max_len = 0
    for line in open(path):
        if line.startswith("Question_ID"):
            count_question+=1

        if line.startswith("Answer_to_Question_ID"):
            count_answer+=1

        if line.strip()=="":
            if len(sentence) > 0:
                #print(sentence)
                output_line = " ".join(w[0] for w in sentence)
                #print(output_line)
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
                    sentence = []
                    continue
                else:
                    #print(output_line)
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

            if parameters['segmentation_only']:
                if gold_label!="O":
                    # print(gold_label)
                    gold_label_prefix=gold_label.split("-")[0]
                    gold_label=gold_label_prefix+"-"+"Name"
                    # print(gold_label)
                    # print("updated gold label")

            

           
            raw_label_name=raw_label.replace("B-","").replace("I-","")
            
            word_info=[gold_word, raw_label_name, raw_label, gold_label]
            
            sentence.append(word_info)

    print("------------------------------------------------------------")
    print("Number of questions in ", path, " : ", count_question)
    print("Number of answers in ", path, " : ", count_answer)
    print("Number of sentences in ", path, " : ", len(sentences))
    print("Max len sentences has", max_len, "words")
    print("------------------------------------------------------------")
    return sentences
                
def load_sentences_so_w_pred(path_main_file, path_segmenter_pred_file,  lower, zeros, merge_tag,set_of_selected_tags):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    count_question=0
    count_answer=0
    max_len = 0

    if merge_tag:
        path=Merge_Label(path_main_file)
    sentences = [] #list of sentences

    sentence = [] #list of words in the current sentence in formate each word list looks like [word, markdow tag name, mark down tag, NER tag]

    for line in open(path):
        if line.startswith("Question_ID"):
            count_question+=1

        if line.startswith("Answer_to_Question_ID"):
            count_answer+=1

        if line.strip()=="":
            if len(sentence) > 0:
                #print(sentence)
                output_line = " ".join(w[0] for w in sentence)
                #print(output_line)
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
                    sentence = []
                    continue
                else:
                    #print(output_line)
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

            if parameters['segmentation_only']:
                if gold_label!="O":
                    # print(gold_label)
                    gold_label_prefix=gold_label.split("-")[0]
                    gold_label=gold_label_prefix+"-"+"Name"
                    # print(gold_label)
                    # print("updated gold label")

            

           
            raw_label_name=raw_label.replace("B-","").replace("I-","")
            
            word_info=[gold_word, raw_label_name, raw_label, gold_label]
            
            sentence.append(word_info)

    


    sentences_preds = []
    sentence_pred = []
    
    for line in open(path_segmenter_pred_file):
        if line.strip()=="":
            if len(sentence_pred) > 0:
                sentences_preds.append(sentence_pred)
                sentence_pred=[]
        else:
            line_values=line.strip().split()
            pred_word= ' '.join(line_values[:-2])
            pred_label=line_values[-1]

            word_info=[pred_word,  pred_label]
            sentence_pred.append(word_info)

    # print(len(sentences_preds),len(sentences))

   



    pred_merged_sentences = []
    for sent_index in range(len(sentences)):
        main_sent = sentences[sent_index]
        pred_sent = sentences_preds[sent_index]
        

        new_sent = []
        new_word_info =[]

        for word_index in range(len(main_sent)):
            [gold_word, raw_label_name, raw_label, gold_label] = main_sent[word_index]
            [pred_word, pred_seg_label] = pred_sent[word_index]
            

            new_word_info = [gold_word, raw_label_name, raw_label,  pred_seg_label, gold_label]
            new_sent.append(new_word_info)

        if len(new_sent)>0:
            pred_merged_sentences.append(new_sent)





    print("------------------------------------------------------------")
    print("Number of questions in ", path, " : ", count_question)
    print("Number of answers in ", path, " : ", count_answer)
    print("Number of sentences in ", path, " : ", len(sentences))
    print("Number of sentences after merging : " , len(pred_merged_sentences))
    print("Max len sentences has", max_len, "words")
    print("------------------------------------------------------------")
    return pred_merged_sentences
                


def load_sentences_conll(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """

    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        #print("prev tags: ",tags)
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag

        else:
            raise Exception('Unknown tagging scheme!')
        # tags = [w[-1] for w in s]
        # print("new tags: ",tags)



def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words) #dict with word frequency
    # print(dico)

    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=3} #prune words which has occureced less than 3 times
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))

    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<PAD>'] = 10000000
    # dico[';'] = 0
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[model.START_TAG] = -1
    dico[model.STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    # print(dico)
    return dico, tag_to_id, id_to_tag

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def hand_features_to_idx(sentences):
    hand_to_idx = []
    count = 0
    for s in sentences:
        hand_to_idx.append(list(range(count, count + len(s))))
        count += len(s)

    return(hand_to_idx)



def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }

def seg_pred_to_idx(sentence):
    # print(type(sentence))
    seg_pred_ids = []
    for word_iter in range(len(sentence)):
        word_info=sentence[word_iter]
        raw_label=word_info[-2]
        if raw_label[0]=='O':
            seg_pred_ids.append(0)
        else:
            seg_pred_ids.append(1)
    return seg_pred_ids


def seg_pred_to_idx_prev(sentence):
    
    seg_pred_ids = []
    for word_iter in range(len(sentence)):
        word_info=sentence[word_iter]
        pred_label=word_info[-1]

        if pred_label=='O':
            seg_pred_ids.append(0)
        else:
            seg_pred_ids.append(1)
        # elif pred_label.startswith("B"):
        #     code_pred_ids.append(1)
        # elif pred_label.startswith("I"):
        #     code_pred_ids.append(2)

    return seg_pred_ids


def ctc_pred_to_idx(sentence, ctc_pred_dict):
    
    ctc_pred_ids = []
    for word_iter in range(len(sentence)):
        word =sentence[word_iter][0]
        if word in ctc_pred_dict:
            ctc_pred_ids.append(int(ctc_pred_dict[word]))
        else:
            ctc_pred_ids.append(0)

    # print(ctc_pred_ids)
    return ctc_pred_ids

def ner_pred_to_idx(sentence, tag_to_id):
    
    ner_pred_ids = []
    for word_iter in range(len(sentence)):
        word_info=sentence[word_iter]
        pred_label=word_info[3]

        pred_label_id = tag_to_id[pred_label]
        ner_pred_ids.append(pred_label_id)

    return ner_pred_ids


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, ctc_pred_dict, lower=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    hands = hand_features_to_idx(sentences)
    
    for i, s in enumerate(sentences):
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        hand = hands[i]
        seg_pred_ids=seg_pred_to_idx(s)
        # seg_pred_ids = seg_pred_to_idx(s)
        ctc_pred_ids = ctc_pred_to_idx(s, ctc_pred_dict)

        

        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
            'seg_pred': seg_pred_ids, #seg pred
            'ctc_pred':ctc_pred_ids,
            'handcrafted': hand
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    #Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    pretrained = []
    for line in codecs.open(ext_emb_path, 'r', 'utf-8'):
        if len(ext_emb_path) > 0:
            try:
                pretrained.append(line.rstrip().split()[0].strip())
            except IndexError:
                continue
    pretrained = set(pretrained)
    for word in words:
        if word not in dictionary and any(x in pretrained for x in [word,word.lower(),re.sub('\d', '0', word.lower())]):
            dictionary[word] = 0 #add the word from dev & test pretrained embedding with 0 freq

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding

    #JT: commented_below : as adding all words from embedding throws CUDA runtime errors
    # if words is None:
    #     for word in pretrained:
    #         if word not in dictionary:
    #             dictionary[word] = 0 #add the word from pretrained embedding with 0 freq
    # else:
    #     for word in words:
    #         if any(x in pretrained for x in [
    #             word,
    #             word.lower(),
    #             re.sub('\d', '0', word.lower())
    #         ]) and word not in dictionary:
    #             dictionary[word] = 0 #add the word from pretrained embedding with 0 freq

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def pad_seq(seq, max_length, PAD_token=0):
    # add pads
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def get_batch(start, batch_size, datas, singletons=[]):
    input_seqs = []
    target_seqs = []
    chars2_seqs = []

    for data in datas[start:start+batch_size]:
        # pair is chosen from pairs randomly
        words = []
        for word in data['words']:
            if word in singletons and np.random.uniform() < 0.5:
                words.append(1)
            else:
                words.append(word)
        input_seqs.append(data['words'])
        target_seqs.append(data['tags'])
        chars2_seqs.append(data['chars'])

    if input_seqs == []:
        return [], [], [], [], [], []
    seq_pairs = sorted(zip(input_seqs, target_seqs, chars2_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, chars2_seqs = zip(*seq_pairs)

    chars2_seqs_lengths = []
    chars2_seqs_padded = []
    for chars2 in chars2_seqs:
        chars2_lengths = [len(c) for c in chars2]
        chars2_padded = [pad_seq(c, max(chars2_lengths)) for c in chars2]
        chars2_seqs_padded.append(chars2_padded)
        chars2_seqs_lengths.append(chars2_lengths)

    input_lengths = [len(s) for s in input_seqs]
    # input_padded is batch * max_length
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    assert target_lengths == input_lengths
    # target_padded is batch * max_length
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # var is max_length * batch_size
    # input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    # target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    #
    # if use_gpu:
    #     input_var = input_var.cuda()
    #     target_var = target_var.cuda()

    return input_padded, input_lengths, target_padded, target_lengths, chars2_seqs_padded, chars2_seqs_lengths


def random_batch(batch_size, train_data, singletons=[]):
    input_seqs = []
    target_seqs = []
    chars2_seqs = []


    for i in range(batch_size):
        # pair is chosen from pairs randomly
        data = random.choice(train_data)
        words = []
        for word in data['words']:
            if word in singletons and np.random.uniform() < 0.5:
                words.append(1)
            else:
                words.append(word)
        input_seqs.append(data['words'])
        target_seqs.append(data['tags'])
        chars2_seqs.append(data['chars'])

    seq_pairs = sorted(zip(input_seqs, target_seqs, chars2_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, chars2_seqs = zip(*seq_pairs)

    chars2_seqs_lengths = []
    chars2_seqs_padded = []
    for chars2 in chars2_seqs:
        chars2_lengths = [len(c) for c in chars2]
        chars2_padded = [pad_seq(c, max(chars2_lengths)) for c in chars2]
        chars2_seqs_padded.append(chars2_padded)
        chars2_seqs_lengths.append(chars2_lengths)

    input_lengths = [len(s) for s in input_seqs]
    # input_padded is batch * max_length
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    assert target_lengths == input_lengths
    # target_padded is batch * max_length
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # var is max_length * batch_size
    # input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    # target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    #
    # if use_gpu:
    #     input_var = input_var.cuda()
    #     target_var = target_var.cuda()

    return input_padded, input_lengths, target_padded, target_lengths, chars2_seqs_padded, chars2_seqs_lengths
