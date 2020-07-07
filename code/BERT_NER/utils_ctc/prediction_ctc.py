
import sys
from os.path import join as path_join
from os.path import dirname
from sys import path as sys_path



# assume script in brat tools/ directory, extend path to find sentencesplit.py
sys_path.append(path_join(dirname(__file__), '.'))

sys.path.append('.')

import torch
import argparse
import numpy as np
from features import Features
from sklearn.metrics import *
from torch.autograd import Variable
import rules

import fasttext

from model import NeuralClassifier
from config_ctc import parameters_ctc
from collections import Counter
from torch.optim import lr_scheduler


fasttext_model = fasttext.load_model('/data/jeniya/STACKOVERFLOW_DATA/POST_PROCESSED/fasttext_model/fasttext.bin')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device=='cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

RESOURCES = {
    "train": "/data/jeniya/STACKOVERFLOW_DATA/CTC/data/train_freq.txt",
    "gigaword_word": "/data/jeniya/STACKOVERFLOW_DATA/CTC/data/gigaword_gt_2.bin",
    "gigaword_char": "/data/jeniya/STACKOVERFLOW_DATA/CTC/data/gigaword_char_unique.bin",
    "stackoverflow_char": "/data/jeniya/STACKOVERFLOW_DATA/CTC/data/no_eng_char_uniq.bin",
    "stackoverflow_word": "/data/jeniya/STACKOVERFLOW_DATA/CTC/data/words.bin",
    "cs": "/data/jeniya/STACKOVERFLOW_DATA/CTC/data/sorted_semantic_scholar_words.txt"
}

def eval(predictions, gold_labels, phase):
    # print(predictions)
    print("--------------------",phase,"--------------------")
    precision = round(precision_score(gold_labels, predictions) * 100.0, 4)
    recall = round(recall_score(gold_labels, predictions) * 100.0, 4)
    f1 = round(f1_score(gold_labels, predictions) * 100.0, 4)

    print("P: ", precision, " R:", recall, " F: ", f1)

    print(classification_report(gold_labels, predictions))

    print("-------------------------------------------------")

def get_word_dict_pre_embeds(train_file, test_file):
    word_id=0
    id_to_word={}
    word_to_id={}
    word_to_vec={}

    for line in open(train_file):
        word=line.split()[0]
        if word not in word_to_id:
            word=word.strip()
            
            word_to_id[word]=word_id
            id_to_word[word_id]=word
            word_to_vec[word]=fasttext_model[word]
            word_id+=1




    for line in open(test_file):
        word=line.split()[0]
        if word not in word_to_id:
            word=word.strip()
            
            word_to_id[word]=word_id
            id_to_word[word_id]=word
            word_to_vec[word]=fasttext_model[word]

            word_id+=1



    

   

    vocab_size = len(word_to_id)

    return vocab_size, word_to_id, id_to_word, word_to_vec

def popluate_word_id_from_file(file_name, word_to_id):
    list_of_ids=[]
    for line in open(file_name):
        word=line.split()[0].strip()
        word_one_hot_vec= np.zeros(len(word_to_id))
        word_id=word_to_id[word]
        word_one_hot_vec[word_id]=1.0

        # list_of_ids.append(word_one_hot_vec)
        list_of_ids.append(word_id)

    arr2d = np.array(list_of_ids)
    # print(arr2d.shape)
    return arr2d

def popluate_word_id_from_token(token, word_to_id):
    list_of_ids=[]
    
    word=token.split()[0].strip()
    if word not in word_to_id:
        word= "**UNK**"
        
    word_one_hot_vec= np.zeros(len(word_to_id))
    word_id=word_to_id[word]
    word_one_hot_vec[word_id]=1.0

    # list_of_ids.append(word_one_hot_vec)
    list_of_ids.append(word_id)

    arr2d = np.array(list_of_ids)
    # print(arr2d.shape)
    return arr2d


def get_train_test_word_id(train_file, test_file, word_to_id):
    train_ids=popluate_word_id_from_file(train_file, word_to_id)
    test_ids=popluate_word_id_from_file(test_file, word_to_id)
    

    return train_ids, test_ids

def prediction_on_token_input(ctc_ip_token, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features):
    

    
    ctc_tokens, ctc_features, ctc_labels = features.get_features_from_token(ctc_ip_token, False)
    
    ctc_ids=popluate_word_id_from_token(ctc_ip_token, word_to_id)


    
    ctc_x = Variable(torch.FloatTensor(ctc_features))
    ctc_x_words = Variable(torch.LongTensor(ctc_ids))
    ctc_y = Variable(torch.LongTensor(ctc_labels))

    
    ctc_scores, ctc_preds = ctc_classifier(ctc_features, ctc_x_words)
    preds=[]
    # fp = open("ctc_ops.tsv", "w")
    # fp.write("token"+"\t"+"true_label"+"\t"+"pred_label"+"\t"+"scores"+"\n")
    for tok, gold, pred, sc in zip(ctc_tokens, ctc_labels, ctc_preds, ctc_scores):
        if rules.IS_NUMBER(tok):
            pred=1
        if rules.IS_URL(tok):
            pred=0
        if pred==1:
            # print(tok, pred)
            pred=1
        preds.append(pred)

    # for tok, gold, pred, sc in zip(ctc_tokens, ctc_labels, ctc_preds, ctc_scores):

    #     fp.write(tok + "\t"  + str(pred) + "\n")

    # fp.close()
    # print(preds[0])
    return preds[0]


def prediction_on_file_input(ctc_input_file, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features):
    

    
    ctc_tokens, ctc_features, ctc_labels = features.get_features(ctc_input_file, False)
    
    ctc_ids=popluate_word_id_from_file(ctc_input_file, word_to_id)


    
    ctc_x = Variable(torch.FloatTensor(ctc_features))
    ctc_x_words = Variable(torch.LongTensor(ctc_ids))
    ctc_y = Variable(torch.LongTensor(ctc_labels))

    
    ctc_scores, ctc_preds = ctc_classifier(ctc_features, ctc_x_words)
    preds=[]
    fp = open("ctc_ops.tsv", "w")
    # fp.write("token"+"\t"+"true_label"+"\t"+"pred_label"+"\t"+"scores"+"\n")
    for tok, gold, pred, sc in zip(ctc_tokens, ctc_labels, ctc_preds, ctc_scores):
        if rules.IS_NUMBER(tok):
            pred=1
        if rules.IS_URL(tok):
            pred=0
        if pred==1:
            # print(tok, pred)
            pred=1
        preds.append(pred)

    for tok, gold, pred, sc in zip(ctc_tokens, ctc_labels, ctc_preds, ctc_scores):

        fp.write(tok + "\t"  + str(pred) + "\n")

    fp.close()






def train_ctc_model(train_file, test_file):

    train_file=parameters_ctc['train_file']
    test_file=parameters_ctc['test_file']


    features = Features(RESOURCES)

    train_tokens, train_features, train_labels = features.get_features(train_file, True)
    test_tokens, test_features, test_labels = features.get_features(test_file, False)
    

    vocab_size, word_to_id, id_to_word, word_to_vec = get_word_dict_pre_embeds(train_file, test_file)
    train_ids, test_ids = get_train_test_word_id(train_file, test_file,  word_to_id)


    
    

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (vocab_size, parameters_ctc['word_dim']))

    for word in word_to_vec:
        word_embeds[word_to_id[word]]=word_to_vec[word]

    


    ctc_classifier = NeuralClassifier(len(train_features[0]), max(train_labels) + 1, vocab_size, word_embeds)
    ctc_classifier.to(device)
    


    optimizer = torch.optim.Adam(ctc_classifier.parameters(), lr=parameters_ctc["LR"])
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    
    train_x = Variable(torch.FloatTensor(train_features).to(device))
    train_x_words = Variable(torch.LongTensor(train_ids).to(device))
    train_y = Variable(torch.LongTensor(train_labels).to(device))


    test_x = Variable(torch.FloatTensor(test_features).to(device))
    test_x_words = Variable(torch.LongTensor(test_ids).to(device))
    test_y = Variable(torch.LongTensor(test_labels).to(device))

    

    for epoch in range(parameters_ctc['epochs']):
        loss = ctc_classifier.CrossEntropy(train_features, train_x_words, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_scores,  train_preds = ctc_classifier(train_features, train_x_words)
        

        test_scores, test_preds = ctc_classifier(test_features, test_x_words)
        # eval(test_preds, test_labels, "test")

    return ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features

    

    


    

    








if __name__ == '__main__':

    train_file=parameters_ctc['train_file']
    test_file=parameters_ctc['test_file']

    ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features= train_model(train_file, test_file)
    
    ip_token = "app"
    op_ctc = prediction_on_token_input(ip_token, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features)
    print(op_ctc)

    
    ip_token = "commit-2"
    op_ctc =prediction_on_token_input(ip_token, ctc_classifier, vocab_size, word_to_id, id_to_word, word_to_vec, features)
    print(op_ctc)




