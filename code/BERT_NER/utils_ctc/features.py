import sys
from os.path import join as path_join
from os.path import dirname
from sys import path as sys_path



# assume script in brat tools/ directory, extend path to find sentencesplit.py
sys_path.append(path_join(dirname(__file__), '.'))

sys.path.append('.')



import re
import math
import kenlm
# import enchant
import numpy as np
from binning import GaussianBinner


class Features:
    def __init__(self, resources, n=5):

        self.gigaword_char = kenlm.Model(resources["gigaword_char"])
        self.gigaword_word = kenlm.Model(resources["gigaword_word"])
        self.stackoverflow_char = kenlm.Model(resources["stackoverflow_char"])
        self.stackoverflow_word = kenlm.Model(resources["stackoverflow_word"])

        self.N = n
        self.binner = GaussianBinner(100)
        # self.en = enchant.Dict("en_US")
        # print("Done loading resources")

    def get_feature_vector(self, word):

        fv = list()

        fv.append(self.gigaword_char.score(" ".join(word.lower())))
        fv.append(self.gigaword_word.score(word.lower(), eos=False, bos=False))

        fv.append(self.stackoverflow_char.score(" ".join(word)))
        score = self.stackoverflow_word.score(word)
        fv.append(score)

        fv.append(word.startswith("http") * 1.0)
        # fv.append(("/" in word and all([self.en.check(t) for t in word.split("/") if len(t) > 0])) * 1.0)
        return fv

    def get_features_from_token(self, token, train):
        # print("Start feature extraction")
        words = []
        labels = []
        features = []
       
        label = int(0) 

        
        # label = 0 if label == 2 else 1

        words.append(token)
        labels.append(label)
        features.append(self.get_feature_vector(token))

        # print("Done feature extraction", set(labels))
        features = self.transform_features(features, train)
        return words, features, labels


    def get_features(self, file_name, train):
        # print("Start feature extraction")

        words = []
        labels = []
        features = []
        for line in open(file_name):
            tokens = line.strip().split("\t")
            if (len(tokens))!=2: continue
            label_val = int(tokens[1]) 

            if label_val==3 or label_val==2:
                label=0
            else:
                label=1
            # label = 0 if label == 2 else 1

            words.append(tokens[0])
            labels.append(label)
            features.append(self.get_feature_vector(tokens[0]))


            
        # print("Done feature extraction", set(labels))
        features = self.transform_features(features, train)
        return words, features, labels

    def transform_features(self, features, train_flag):
        features = np.array(features)
        if train_flag:
            self.binner.fit(features, self.N)
        return self.binner.transform(features, self.N)
