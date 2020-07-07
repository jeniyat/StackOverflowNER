import torch
import argparse
import numpy as np
from features import Features
from sklearn.metrics import *
from torch.autograd import Variable
import rules
from config_ctc import parameters_ctc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device=='cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class NeuralClassifier(torch.nn.Module):
    def __init__(self, input_feat_dim, target_label_dim, vocab_size, pre_word_embeds=None):
        super(NeuralClassifier, self).__init__()

        
        hidden_layer_node = parameters_ctc['hidden_layer_1_dim']
        self.Linear_Layer_1=torch.nn.Linear(input_feat_dim, hidden_layer_node)
        
        self.Tanh_Layer=torch.nn.Tanh()
        

        self.Word_Embeds = torch.nn.Embedding(vocab_size, parameters_ctc['word_dim'])
        if pre_word_embeds.any():
            self.Word_Embeds.weight = torch.nn.Parameter(torch.FloatTensor(pre_word_embeds))

        self.Linear_Layer_2=torch.nn.Linear(hidden_layer_node, target_label_dim)
        self.Linear_Layer_2=torch.nn.Linear(hidden_layer_node, hidden_layer_node)

        self.Linear_Layer_3=torch.nn.Linear(hidden_layer_node+parameters_ctc['word_dim'], target_label_dim)

        

        self.Softmax_Layer=torch.nn.Softmax(dim=1)


        self.loss_fn = torch.nn.CrossEntropyLoss()


    def forward(self, features, word_ids):
        scores = self.get_scores(features, word_ids)
        # print(device)

        if torch.cuda.is_available():
            # print("---------------------")
            scores_ = scores.cpu().data.numpy()
        else:
            scores_ = scores.data.numpy()
        predictions = [np.argmax(sc) for sc in scores_]

        
        return scores, predictions

    
    def get_scores(self, features, word_ids):
        features_x = Variable(torch.FloatTensor(features).to(device))
        word_ids=word_ids.to(device)

        liner1_op = self.Linear_Layer_1(features_x)
        tanh_op = self.Tanh_Layer(liner1_op)

        # liner2_op = self.Linear_Layer_2(tanh_op)
        # tanh_op = self.Tanh_Layer(liner2_op)



        word_embeds=self.Word_Embeds(word_ids)

        # print(type(tanh_op))
        # print(tanh_op.size())
        # print(type(word_embeds))
        # print(word_embeds.size())



        features_embed_cat = torch.cat((word_embeds,tanh_op ),dim=1)

        liner3_op=self.Linear_Layer_3(features_embed_cat)

        # print(features_embed_cat.size())

        
        scores = self.Softmax_Layer(liner3_op)

        return scores

    def CrossEntropy(self, features, word_ids, gold_labels):
        # features_x = Variable(torch.FloatTensor(features))
        scores= self.get_scores(features, word_ids)
        loss = self.loss_fn(scores, gold_labels)

        return loss


    def predict(self, features, word_ids):
        scores= self.get_scores(features, word_ids).data.numpy()
        predictions = [np.argmax(sc) for sc in scores]

        return predictions



