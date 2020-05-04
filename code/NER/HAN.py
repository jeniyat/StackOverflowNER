import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

from utils_so import *
from config_so import parameters

torch.backends.cudnn.deterministic = True
torch.manual_seed(parameters["seed"])





class Embeeding_Attn(nn.Module):
  def __init__(self):
    super(Embeeding_Attn, self).__init__()
    
    self.max_len = 3
    self.input_dim = 1824
    self.hidden_dim = 150
    self.bidirectional = True
    self.drop_out_rate = 0.5 

    self.context_vector_size = [parameters['embedding_context_vecotr_size'], 1]
    self.drop = nn.Dropout(p=self.drop_out_rate)

    self.word_GRU = nn.GRU(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               batch_first=True)
    
    self.w_proj = nn.Linear(in_features=2*self.hidden_dim ,out_features=2*self.hidden_dim)

    self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size).float())

    self.softmax = nn.Softmax(dim=1)

    init_gru(self.word_GRU)

  def forward(self,x):
    
    
    x, _ = self.word_GRU(x)
    Hw = torch.tanh(self.w_proj(x))
    w_score = self.softmax(Hw.matmul(self.w_context_vector))
    x = x.mul(w_score)
    x = torch.sum(x, dim=1)
    return x








class Word_Attn(nn.Module):
  def __init__(self):
    super(Word_Attn, self).__init__()
    
    self.max_len = 92
    self.input_dim = 300
    self.hidden_dim = 150
    self.bidirectional = True
    self.drop_out_rate = 0.5 

    self.context_vector_size = [parameters['word_context_vecotr_size'] , 1]
    self.drop = nn.Dropout(p=self.drop_out_rate)

    self.word_GRU = nn.GRU(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               bidirectional=self.bidirectional,
                               batch_first=True)
    
    self.w_proj = nn.Linear(in_features=2*self.hidden_dim ,out_features=2*self.hidden_dim)

    self.w_context_vector = nn.Parameter(torch.randn(self.context_vector_size).float())

    self.softmax = nn.Softmax(dim=1)

    init_gru(self.word_GRU)

  def forward(self,x):
    
    
    x, _ = self.word_GRU(x)
    Hw = torch.tanh(self.w_proj(x))
    w_score = self.softmax(Hw.matmul(self.w_context_vector))
    x = x.mul(w_score)
    # print(x.size())
    return x









