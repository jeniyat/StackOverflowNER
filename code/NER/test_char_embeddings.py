# coding=utf-8

from __future__ import print_function
import optparse
import itertools
from collections import OrderedDict
import torch
import time
import pickle
from torch.optim import lr_scheduler
from torch.autograd import Variable
# import matplotlib.pyplot as plt     #JT: commented it
import sys
import os
import json
import numpy as np
import codecs
# import Visdom     #JT: commented it
# from utils import *
# from loader import *
# from config import opts


# from model_wo_char import BiLSTM_CRF
from model import BiLSTM_CRF


import utils_so as utils    #JT: utils for SO
import loader_so as loader  #JT: loader for SO
from config_so import parameters
from config_so import opts
from utils_so import Sort_Entity_by_Count
import shutil

# from evaluate_so import evaluating
# sys.path.append('../../utility/')
import print_result 
import conlleval_py 
import tolatex


torch.backends.cudnn.deterministic = True
torch.manual_seed(parameters["seed"])

np.random.seed(parameters["seed"])



from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


assert os.path.isfile(parameters["train"])
assert os.path.isfile(parameters["dev"])
assert os.path.isfile(parameters["test"])
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
# assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])







def prepare_train_set_dev_data():

    lower = parameters['lower']
    zeros = parameters['zeros']
    tag_scheme = parameters['tag_scheme']

    #------------------------------------------------------------------
    #------------- prepare the training data --------------------------
    #------------- merge labels and select category specific entities -
    #------------------------------------------------------------------

    input_train_file=utils.Merge_Label(parameters["train"])
    
    Sort_Entity_by_Count(input_train_file,parameters["sorted_entity_list_file_name"])

    with open(parameters["sorted_entity_list_file_name"]) as f:
        sorted_entity_list = json.load(f)

    set_of_selected_tags=[]




    entity_category_code=parameters["entity_category_code"]
    entity_category_human_language=parameters["entity_category_human_language"]


    set_of_selected_tags.extend(sorted_entity_list[0:-6])

    if parameters['entity_category']=='code':
        for entity in entity_category_human_language:
            if entity in entity_category_human_language and entity in set_of_selected_tags:
                set_of_selected_tags.remove(entity)

        

    if parameters['entity_category']=='human_lang':
        for entity in entity_category_code:
            if entity in entity_category_code and entity in set_of_selected_tags:
                set_of_selected_tags.remove(entity)

        if 'Algorithm' not in set_of_selected_tags:
            set_of_selected_tags.append('Algorithm')

    if parameters['entity_category']=='all':
        if 'Algorithm' not in set_of_selected_tags:
            set_of_selected_tags.append('Algorithm')




    print("set_of_selected_tags: ", set_of_selected_tags)

    merge_tags=parameters['merge_tags']


    train_sentences = loader.load_sentences_so(parameters["train"], lower, zeros,merge_tags, set_of_selected_tags)
    dev_sentences = loader.load_sentences_so(parameters["dev"], lower, zeros,merge_tags, set_of_selected_tags)
    test_sentences = loader.load_sentences_so(parameters["test"], lower, zeros,merge_tags, set_of_selected_tags)


    loader.update_tag_scheme(train_sentences, tag_scheme)
    loader.update_tag_scheme(dev_sentences, tag_scheme)
    loader.update_tag_scheme(test_sentences, tag_scheme)


    dico_words_train = loader.word_mapping(train_sentences, lower)[0]
    dico_chars, char_to_id, id_to_char = loader.char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = loader.tag_mapping(train_sentences)

    #------------------------------------------------------------------------------------------------------------
    #------------- based on parameters setting(should be set by command line argutments) ------------------------
    #------------- load pretrained word embeddings --------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------


    if parameters['all_emb']:
        all_dev_test_words=[w[0][0] for w in dev_sentences+test_sentences]
    else:
        all_dev_test_words = []


    if parameters['use_pre_emb']:
        dico_words, word_to_id, id_to_word = loader.augment_with_pretrained(
                dico_words_train.copy(),
                parameters['pre_emb'],
                all_dev_test_words
            )
    else:
        dico_words = dico_words_train
        word_to_id, id_to_word = loader.create_mapping(dico_words_train.copy())

    train_data = loader.prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, lower)
    dev_data = loader.prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, lower)
    test_data = loader.prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, lower)


    all_word_embeds = {}
    if parameters['use_pre_emb']:
        for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
            # print(line)
            s = line.strip().split()
            if len(s) == parameters['word_dim'] + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))

    if parameters['use_pre_emb']:
        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]


    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


    mapping_file = parameters["models_path"]+'/mapping.pkl'

    with open(mapping_file, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'id_to_word': id_to_word,
            'tag_to_id': tag_to_id,
            'char_to_id': char_to_id,
            'id_to_char': id_to_char,
            'parameters': parameters,
            'word_embeds': word_embeds
        }
        pickle.dump(mappings, f, protocol=4)


    return train_data, dev_data, test_data, word_to_id, id_to_word,  tag_to_id, id_to_tag, char_to_id, id_to_char, word_embeds





# vis = visdom.Visdom() #JT: no need of visualization for now
# sys.stdout.flush()


def evaluating(model, datas, best_F, epoch_count, phase_name):
    
    print("-----------------------------------")
    print("now evaluating: ",phase_name)
    print("-----------------------------------")
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))

    iter_count=0
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']

        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')

    predf = parameters["eval_temp"] + '/pred.' + name +"_"+str(epoch_count)
    scoref = parameters["eval_temp"] + '/score.' + name+"_"+str(epoch_count)


    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    eval_result = conlleval_py.evaluate_conll_file(inputFile=predf)

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)


    #-------------------------------------------------------------------------------------------------
    #--------------- only print the performnace on dev/test set. do not print for train set ----------
    #-------------------------------------------------------------------------------------------------

    if phase_name=="dev" or phase_name=="test":
        print_result.print_result(eval_result, epoch_count, parameters["sorted_entity_list_file_name"], parameters["entity_category_code"], parameters["entity_category_human_language"])
        print("-----------------------------------")
        over_all_p=eval_result['overall']['P']
        over_all_r=eval_result['overall']['R']
        over_all_f1=eval_result['overall']['F1']
        op_line = phase_name+ ": epoch: "+str(epoch_count) +" P: "+ str(over_all_p)+" R: "+str(over_all_r)+" F1: "+str(over_all_f1)+"\n"
        # fout_per_epoch.write(op_line)
        # fout_per_epoch.flush()
    return best_F, new_F, save


def save_char_embed(sentence_words, char_embed_dict, char_embed_vectors):
    # print(sentence_words)
    # print(char_embed_dict)
    # print(char_embed_vectors)
    

    for sent_iter in range(len(sentence_words)):
        word = sentence_words[sent_iter]
        word_char_vector = char_embed_vectors[sent_iter]
        char_embed_dict[word]=word_char_vector
        # print(word, word_char_vector)

    return char_embed_dict

def train_model(model, step_lr_scheduler, optimizer, train_data, dev_data, test_data):
    char_embed_dict = {}

    losses = []
    loss = 0.0
    best_dev_F = -1.0
    best_test_F = -1.0
    best_train_F = -1.0
    all_F = [[0, 0, 0]]
    plot_every = 10
    eval_every = 20
    count = 0

    model.train(True)
    for epoch in range(1, parameters["epochs"]+1):
        
        print("---------epoch count: ", epoch)
        for i, index in enumerate(np.random.permutation(len(train_data))):
            
            tr = time.time()
            count += 1
            data = train_data[index]
            #what is the data instance looks like"
            # 'str_words': ['Trial', 'and', 'error', 'seems', 'a', 'very', 'dumb', '(', 'and', 'annoying', ')', 'approach', 'to', 'solve', 'this', 'problem', '.'], 
            #'words': [1, 9, 76, 179, 7, 215, 1, 26, 9, 1, 29, 332, 4, 310, 15, 64, 3], 
            #'chars': [[26, 8, 5, 4, 10], [4, 6, 11], [1, 8, 8, 3, 8], [7, 1, 1, 14, 7], [4], [22, 1, 8, 17], [11, 13, 14, 21], [35], [4, 6, 11], [4, 6, 6, 3, 17, 5, 6, 16], [34], [4, 15, 15, 8, 3, 4, 12, 9], [2, 3], [7, 3, 10, 22, 1], [2, 9, 5, 7], [15, 8, 3, 21, 10, 1, 14], [20]], 
            #'caps': [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            #'tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'handcrafted': [28052, 28053, 28054, 28055, 28056, 28057, 28058, 28059, 28060, 28061, 28062, 28063, 28064, 28065, 28066, 28067, 28068]
            model.zero_grad()

            sentence_in = data['words']
            
            tags = data['tags']
            chars2 = data['chars']

            # print(data)

            sentence_in = Variable(torch.LongTensor(sentence_in))

            ######### char lstm
            if parameters['char_mode'] == 'LSTM':
                chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
                d = {}
                for i, ci in enumerate(chars2):
                    for j, cj in enumerate(chars2_sorted):
                        if ci == cj and not j in d and not i in d.values():
                            d[j] = i
                            continue
                chars2_length = [len(c) for c in chars2_sorted]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
                for i, c in enumerate(chars2_sorted):
                    chars2_mask[i, :chars2_length[i]] = c
                chars2_mask = Variable(torch.LongTensor(chars2_mask))

            # ######## char cnn
            if parameters['char_mode'] == 'CNN':
                d = {}
                chars2_length = [len(c) for c in chars2]
                char_maxl = max(chars2_length)
                chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                for i, c in enumerate(chars2):
                    chars2_mask[i, :chars2_length[i]] = c
                # print(chars2_mask)
                chars2_mask = Variable(torch.LongTensor(chars2_mask))


            targets = torch.LongTensor(tags)
            caps = Variable(torch.LongTensor(data['caps']))
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d)
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d)
            # loss += neg_log_likelihood.data[0] / len(data['words'])

            if use_gpu:
                char_embed_op = model.get_char_embedding(sentence_in.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d).clone().data.cpu().numpy()
            else:
                char_embed_op = model.get_char_embedding(sentence_in, chars2_mask, caps, chars2_length, d).clone().data.cpu().numpy()

            char_embed_dict = save_char_embed( data['str_words'], char_embed_dict, char_embed_op)

            # print(char_embed_op)
            loss += neg_log_likelihood.data.item() / len(data['words']) #JT : data[0]> data.item()
            neg_log_likelihood.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) #JT
            optimizer.step()

            

            


            if count % len(train_data) == 0:
                utils.adjust_learning_rate(optimizer, lr=learning_rate/(1+0.05*count/len(train_data)))

        #JT: evaluate after 1 epoch
        model.train(False)

        best_train_F, new_train_F, _ = evaluating(model, train_data,  best_train_F,  epoch, "train")
        best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F, epoch, "dev") 
        if save:
            torch.save(model, model_name)
        best_test_F, new_test_F = 0, 0
        all_F.append([new_train_F, new_dev_F, new_test_F])
        step_lr_scheduler.step()

        
        
        # word_embeding_weights=model.word_embeds.weight.data.cpu().numpy()
        # print("type(word_embeding_weights): ", type(word_embeding_weights))
        # print("shape word_embeding_weights: ", word_embeding_weights.shape)
        # print("shape word_embeding_weights: ", model.word_embeds.weight.data.size())
        # print("shape word_embeding_weights: ", model.word_embeds.weight.data[0])

        #-------------------------------------------------------------------------------------------------
        #--------------------- save model for each epoch, after finding the optimal epoch ----------------
        #--------------------- save model from last epoch only -------------------------------------------
        #-------------------------------------------------------------------------------------------------

        PATH=parameters["models_path"]+"/model_epoch."+str(epoch)
        torch.save(model, PATH)
        model.train(True)


    return char_embed_dict

    


    

def get_char_embedding_dict(datas, model, parameters):
    char_embed_dict={}
    for data in datas:
        sentence_in = data['words']
        chars2 = data['chars']
        sentence_in = Variable(torch.LongTensor(sentence_in))
        if parameters['char_mode'] == 'LSTM':
            chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
            d = {}
            for i, ci in enumerate(chars2):
                for j, cj in enumerate(chars2_sorted):
                    if ci == cj and not j in d and not i in d.values():
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in chars2_sorted]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
            for i, c in enumerate(chars2_sorted):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        # ######## char cnn
        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            # print(chars2_mask)
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        caps = Variable(torch.LongTensor(data['caps']))
        
        if use_gpu:
            char_embed_op = model.get_char_embedding(sentence_in.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d).clone().data.cpu().numpy()
        else:
            char_embed_op = model.get_char_embedding(sentence_in, chars2_mask, caps, chars2_length, d).clone().data.cpu().numpy()

        char_embed_dict = save_char_embed( data['str_words'], char_embed_dict, char_embed_op)

    
    print(len(char_embed_dict.keys()))
    return char_embed_dict
   


def test_dev_oov_char_embedding(train_data, dev_data, model, parameters, train_data_char_embed_dict):
    dev_data_char_embed_dict=get_char_embedding_dict(dev_data, model, parameters)
    # train_data_char_embed_dict=get_char_embedding_dict(train_data, model, parameters)
    print(len(train_data_char_embed_dict.keys()))


    oov_list=set(dev_data_char_embed_dict.keys()) - set((train_data_char_embed_dict.keys()))
    fout=open("oov_words.txt",'w')
    fout.write("\n".join(oov_list))
    fout.close()


    outfile=open("train_vocab.txt",'w')
    outfile.write("\n".join(train_data_char_embed_dict.keys()))
    outfile.close()
    # print(oov_list)
    # print(len(oov_list))
    return dev_data_char_embed_dict

def find_similar_word(train_char_embedding, word_char_embed, limit = 11):

    
    similarity_dict=defaultdict(float)
    for w in train_char_embedding:
        try:
            char_embedding= train_char_embedding[w]
            similarity = cosine_similarity(word_char_embed.reshape(1,-1), char_embedding.reshape(1,-1))
            similarity_dict[w]=similarity
        except Exception as e:
            continue
        

    count = 0
    for w in sorted(similarity_dict, key=similarity_dict.get, reverse=True):
        print(w, similarity_dict[w])
        count+=1
        if count==limit:
            break



if __name__ == '__main__':
    eval_script= parameters["eval_script"]

    eval_temp= parameters["eval_temp"]
    try:
        shutil.rmtree(eval_temp)
    except Exception as e:
        pass


    fout_per_epoch = open(parameters["perf_per_epoch_file"],'w')
    fout_per_epoch.close()

    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)
    if not os.path.exists(parameters["models_path"]):
        os.makedirs(parameters["models_path"])

    
    train_data, dev_data, test_data, word_to_id, id_to_word,  tag_to_id, id_to_tag, char_to_id, id_to_char, word_embeds=prepare_train_set_dev_data()


    use_gpu = parameters['use_gpu']
    gpu_id = parameters["gpu_id"]

    name = parameters['name']
    model_name = parameters["models_path"] + name #get_name(parameters)
    tmp_model = model_name + '.tmp'

    model = BiLSTM_CRF(vocab_size=len(word_to_id),
                       tag_to_ix=tag_to_id,
                       embedding_dim=parameters['word_dim'],
                       hidden_dim=parameters['word_lstm_dim'],
                       use_gpu=use_gpu,
                       char_to_ix=char_to_id,
                       pre_word_embeds=word_embeds,
                       use_crf=parameters['crf'],
                       char_mode=parameters['char_mode'],
                       # n_cap=4,
                       # cap_embedding_dim=10
                       )
    model=torch.load('model_epoch.24',map_location='cpu')

    with open('char_embed_dict_6.json', 'rb') as handle:
        train_data_char_embed_dict = pickle.load(handle)

    with open('mapping.pkl', 'rb') as handle:
        mappings = pickle.load(handle)
    
    # print(char_embed_dict)
    # print(mappings)
    word_to_id=mappings['word_to_id']
    tag_to_id=mappings['tag_to_id']
    char_to_id=mappings['char_to_id']
    parameters=mappings['parameters']
    # print(char_to_id)



    dev_data_char_embed_dict = test_dev_oov_char_embedding(train_data, dev_data, model, parameters, train_data_char_embed_dict)

    selected_word = "4.1.2"
    # selected_word_char_emebd=train_data_char_embed_dict[selected_word]
    selected_word_char_emebd=dev_data_char_embed_dict[selected_word]
    find_similar_word(dev_data_char_embed_dict, selected_word_char_emebd)

    
    
    

    
    

