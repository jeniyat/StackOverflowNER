import torch
import torch.autograd as autograd
from torch.autograd import Variable



import utils_so as utils 
from utils_so import *

from config_so import parameters

torch.backends.cudnn.deterministic = True
torch.manual_seed(parameters["seed"])


from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder


from HAN import *
START_TAG = '<START>'
STOP_TAG = '<STOP>'


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def _align_word(input_matrix, word_pos_list=[1]):
    """
        To change presentation of the batch.
    Args:
        input_matrix (autograd.Variable):  The the presentation of the word in the sentence. [sentence_num, sentence_embedding]
        word_pos_list (list): The list contains the position of the word in the current sentence.
        
    Returns:
        new_matrix (torch.FloatTensor): The aligned matrix, and its each row is one sentence in the passage.
                                        [passage_num, max_len, embedding_size]
    """
    # assert isinstance(input_matrix, torch.autograd.Variable), 'The input object must be Variable'

    embedding_size = input_matrix.shape[-1]      # To get the embedding size of the sentence
    number_of_words = len(word_pos_list)                  # To get the number of the sentences
    # if sent_max is not None:
    #     max_len = sent_max
    # else:
    #     max_len = torch.max(sent_num)
    max_len=1
    new_matrix = autograd.Variable(torch.zeros(number_of_words, max_len, embedding_size))
    init_index = 0
    for index, length in enumerate(word_pos_list):
        end_index = init_index + length

        # temp_matrix
        temp_matrix = input_matrix[init_index:end_index, :]      # To get one passage sentence embedding.
        if temp_matrix.shape[0] > max_len:
            temp_matrix = temp_matrix[:max_len]
        new_matrix[index, -length:, :] = temp_matrix

        # update the init_index of the input matrix
        init_index = length
    return new_matrix





    
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, freq_embed_dim, markdown_embed_dim,seg_pred_embed_dim, hidden_dim, char_lstm_dim=25,
                 char_to_ix=None, pre_word_embeds=None, word_freq_embeds=None, word_ner_pred_embeds=None, word_seg_pred_embeds=None, word_markdown_embeds=None, word_ctc_pred_embeds=None, char_embedding_dim=25, use_gpu=False,
                 n_cap=None, cap_embedding_dim=None, use_crf=True, char_mode='CNN'):
        super(BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.n_cap = n_cap
        self.cap_embedding_dim = cap_embedding_dim
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_lstm_dim
        self.char_mode = char_mode

        self.embed_attn = Embeeding_Attn()
        self.word_attn = Word_Attn()



        

        #init elmo if use_elmo is set to True(1)
        self.use_elmo = parameters['use_elmo']

        if self.use_elmo:
            options_file = parameters["elmo_options"]
            weight_file = parameters["elmo_weight"]

            self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
            self.elmo_2 = ElmoEmbedder(options_file, weight_file)
            


        print('char_mode: %s, out_channels: %d, hidden_dim: %d, ' % (char_mode, char_lstm_dim, hidden_dim))
        # self.lstm=nn.LSTM(self.get_lstm_input_dim(),hidden_dim, bidirectional=True)
        if parameters['use_han']:
            self.lstm=nn.LSTM(300,hidden_dim, bidirectional=True)
        else:
            self.lstm=nn.LSTM(1824,hidden_dim, bidirectional=True)
        
        
        

        if self.use_elmo:
            
            if parameters['use_elmo_w_char']:
                if self.n_cap and self.cap_embedding_dim:
                    self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
                    init_embedding(self.cap_embeds.weight)

                if char_embedding_dim is not None:
                    self.char_lstm_dim = char_lstm_dim
                    self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
                    init_embedding(self.char_embeds.weight)

                    if self.char_mode == 'LSTM':
                        self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                        init_lstm(self.char_lstm)
                    if self.char_mode == 'CNN':
                        self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))

                    

           

        else:
            if self.n_cap and self.cap_embedding_dim:
                self.cap_embeds = nn.Embedding(self.n_cap, self.cap_embedding_dim)
                init_embedding(self.cap_embeds.weight)

            if char_embedding_dim is not None:
                self.char_lstm_dim = char_lstm_dim
                self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
                init_embedding(self.char_embeds.weight)
                
                if self.char_mode == 'LSTM':
                    self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True)
                    init_lstm(self.char_lstm)
                if self.char_mode == 'CNN':
                    self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))

            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
            if pre_word_embeds is not None:
                self.pre_word_embeds = True
                self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
            else:
                self.pre_word_embeds = False

            
            
            


        self.dropout = nn.Dropout(parameters['dropout'])

        #----------adding frequency embedding--------
        self.freq_embeds = nn.Embedding(vocab_size, freq_embed_dim)
        if word_freq_embeds is not None:
            self.word_freq_embeds = True
            self.freq_embeds.weight = nn.Parameter(torch.FloatTensor(word_freq_embeds))
        else:
            self.word_freq_embeds = False

        #----------adding markdown embedding--------
        # self.markdown_embeds = nn.Embedding(parameters['markdown_count'], markdown_embed_dim)
        # if word_markdown_embeds is not None:
        #     self.use_markdown_embed = True
        #     self.markdown_embeds.weight = nn.Parameter(torch.FloatTensor(word_markdown_embeds))
        # else:
        #     self.use_markdown_embed = False

        #----------adding segmentation embedding--------
        self.seg_embeds = nn.Embedding(parameters['segmentation_count'], parameters['segmentation_dim'])
        if word_seg_pred_embeds is not None:
            self.use_seg_pred_embed = True
            self.seg_embeds.weight = nn.Parameter(torch.FloatTensor(word_seg_pred_embeds))
        else:
            self.use_seg_pred_embed = False


        #----------adding ctc prediction embedding--------
        self.ctc_pred_embeds = nn.Embedding(parameters['code_recognizer_count'], parameters['code_recognizer_dim'])
        if word_ctc_pred_embeds is not None:
            self.use_ctc_pred_embed = True
            self.ctc_pred_embeds.weight = nn.Parameter(torch.FloatTensor(word_ctc_pred_embeds))
        else:
            self.use_ctc_pred_embed = False


        #----------adding ner prediction embedding--------
        # self.ner_pred_embeds = nn.Embedding(parameters['ner_pred_count'], parameters['ner_pred_dim'])
        # if word_ner_pred_embeds is not None:
        #     self.use_ner_pred_embed = True
        #     self.ner_pred_embeds.weight = nn.Parameter(torch.FloatTensor(word_ner_pred_embeds))
        # else:
        #     self.use_ner_pred_embed = False


        init_lstm(self.lstm)
        # init_lstm(self.lstm_2)
        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)

        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))
            self.transitions.data[tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    


    
    


    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score    

    def get_char_embedding(self, sentence, chars2, caps, chars2_length, d):

        if self.char_mode == 'LSTM':
            # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars2.size(0))
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            # chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_cnn_out3 = nn.functional.relu(self.char_cnn3(chars_embeds))   
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                 kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

        # t = self.hw_gate(chars_embeds)
        # g = nn.functional.sigmoid(t)
        # h = nn.functional.relu(self.hw_trans(chars_embeds))
        # chars_embeds = g * h + (1 - g) * chars_embeds

        # embeds = self.word_embeds(sentence)
        # # word_emebd_parameters = m.weight.numpy()
        # if self.n_cap and self.cap_embedding_dim:
        #     cap_embedding = self.cap_embeds(caps)

        # if self.n_cap and self.cap_embedding_dim:
        #     embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        # else:
        #     embeds = torch.cat((embeds, chars_embeds), 1)

        # embeds = embeds.unsqueeze(1)
        # embeds = self.dropout(embeds)
        # lstm_out, _ = self.lstm(embeds)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        # lstm_out = self.dropout(lstm_out)
        # lstm_feats = self.hidden2tag(lstm_out)
        return chars_embeds

    def _get_lstm_features_w_elmo_and_char(self, sentence_words, sentence,markdown, chars2, caps, chars2_length, d):
        if self.char_mode == 'LSTM':
            # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars2.size(0))
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            # chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_cnn_out3 = nn.functional.relu(self.char_cnn3(chars_embeds))   
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                 kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)


        character_ids = batch_to_ids([sentence_words])
        if self.use_gpu:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)
        embeddings = embeddings['elmo_representations'][0]
        embeds = embeddings[0]
       
        
        
        if self.use_gpu:
            embeds=embeds.cuda()

        embeds = torch.cat((embeds, chars_embeds), 1)
        
        
       

       


        embeds = embeds.unsqueeze(1)
        # print(embeds.size())
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence_words), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def apply_attention(self, elmo_embeds, seg_embeds, ctc_embeds):
        word_tensor_list = []
        word_pos_list=[]
        sent_len =elmo_embeds.size()[0]

        for index in range(sent_len):
            elmo_rep = elmo_embeds[index]
            ctc_rep = ctc_embeds[index]
            seg_rep = seg_embeds[index]
            comb_rep = torch.cat((elmo_rep, ctc_rep, seg_rep)).view(1, 1, -1)
            # print(comb_rep.size())
            attentive_rep=self.embed_attn(comb_rep)
            word_tensor_list.append(attentive_rep)
            word_pos_list.append(index+1)

        word_tensor =  torch.stack(word_tensor_list)
        if self.use_gpu:
            word_tensor=word_tensor.cuda()

        
        x = _align_word(word_tensor, word_pos_list)
        if self.use_gpu:
            x=x.cuda()
        y = self.word_attn(x)
        # print("y size: ", y.size())
        return y


    def _get_lstm_features_w_elmo(self, sentence_words, sentence, seg_pred, ctc_pred):
        

        character_ids = batch_to_ids([sentence_words])
        if self.use_gpu:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)
        embeddings = embeddings['elmo_representations'][0]
        embeds = embeddings[0]
        # print("embeds size: ",embeds.size())
        # bilm_layer_1 = embeddings[1]
        # bilm_layer_2 = embeddings[2]
        if self.use_gpu:
            embeds=embeds.cuda()
            elmo_embeds=embeds.cuda()

        if parameters['use_freq_vector']:
            frequency_embeddings = self.freq_embeds(sentence)
            embeds = torch.cat((embeds, frequency_embeddings), 0)
        
        
        
        # # print("embeds size: ",embeds.size())
        # # print("embeds size w current : ", embeds.size())
        # # embeds = embeds.unsqueeze(1)
        # # print("embeds size w current : ", embeds.size())


        # 
        if parameters['use_segmentation_vector'] :
            segment_embeddings = self.seg_embeds(seg_pred)
            embeds = torch.cat((embeds, segment_embeddings), 1)
        # print("markdown_embeddings.unsqueeze(1) size:", markdown_embeddings.unsqueeze(1).size())

        if parameters['use_code_recognizer_vector']:
            ctc_pred_embeddings = self.ctc_pred_embeds(ctc_pred)
            embeds = torch.cat((embeds, ctc_pred_embeddings), 1)
        
        

        if parameters['use_han']:
            attentive_word_embeds = self.apply_attention(elmo_embeds, segment_embeddings, ctc_pred_embeddings)
            embeds=attentive_word_embeds

        
        else:
            embeds = embeds.unsqueeze(1)


        embeds = self.dropout(embeds)
        

        # embeds = self.dropout(merged_embeds)
        lstm_out, _ = self.lstm(embeds)
        #lstm_out, _ = self.lstm_2(lstm_out)
        lstm_out = lstm_out.view(len(sentence_words), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats



    def _get_lstm_features_w_elmo_prev(self, sentence_words, sentence, markdown, seg_pred):
        # print(sentence_words)
        # print(sentence)
        # print(markdown)
        # print(seg_pred)

        # print(len(sentence_words))
        # embeddings =self.elmo_2.embed_sentence(sentence_words)
        # char_layer_embed = embeddings[0]
        # bilm_layer_1 = embeddings[1]
        # bilm_layer_2 = embeddings[2]

        # embeds=torch.from_numpy(bilm_layer_2).float()
        # if self.use_gpu:
        #     embeds=embeds.cuda()
        # print("embeds size before: ", embeds.size())
        # embeds = embeds.unsqueeze(1)
        # print("embeds size before: ", embeds.size())



        character_ids = batch_to_ids([sentence_words])
        if self.use_gpu:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)
        embeddings = embeddings['elmo_representations'][0]
        embeds = embeddings[0]
        # print("embeds size: ",embeds.size())
        # bilm_layer_1 = embeddings[1]
        # bilm_layer_2 = embeddings[2]
        if self.use_gpu:
            embeds=embeds.cuda()
            elmo_embeds=embeds.cuda()

        if parameters['use_freq_vector']:
            frequency_embeddings = self.freq_embeds(sentence)
            embeds = torch.cat((embeds, frequency_embeddings), 0)
        
        
        
        # print("embeds size: ",embeds.size())
        # print("embeds size w current : ", embeds.size())
        # embeds = embeds.unsqueeze(1)
        # print("embeds size w current : ", embeds.size())


        # 
        if parameters['use_markdown_vector'] :
            markdown_embeddings = self.markdown_embeds(markdown)
            embeds = torch.cat((embeds, markdown_embeddings), 1)

        if parameters['use_segmentation_vector']:
            seg_pred_embeddings = self.seg_pred_embeds(seg_pred)
            embeds = torch.cat((embeds, seg_pred_embeddings), 1)

        attentive_word_embeds = self.apply_attention(elmo_embeds, markdown_embeddings, seg_pred_embeddings)

        print("attentive_word_embeds.size(): ",attentive_word_embeds.size())
        print("embeds.size(): ", embeds.size())

        embeds = embeds.unsqueeze(1)
        print("embeds.unsqueeze(1): ",embeds.size())
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        #lstm_out, _ = self.lstm_2(lstm_out)
        lstm_out = lstm_out.view(len(sentence_words), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


        

    def _get_lstm_features(self, sentence, markdown, chars2, caps, chars2_length, d):

        if self.char_mode == 'LSTM':
            # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars2.size(0))
            chars_embeds = self.char_embeds(chars2).transpose(0, 1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
            lstm_out, _ = self.char_lstm(packed)
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.cuda()
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = torch.cat((outputs[i, index-1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
            chars_embeds = chars_embeds_temp.clone()
            for i in range(chars_embeds.size(0)):
                chars_embeds[d[i]] = chars_embeds_temp[i]

        if self.char_mode == 'CNN':
            chars_embeds = self.char_embeds(chars2).unsqueeze(1)
            # chars_cnn_out3 = self.char_cnn3(chars_embeds)
            chars_cnn_out3 = nn.functional.relu(self.char_cnn3(chars_embeds))   
            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                                                 kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

        # t = self.hw_gate(chars_embeds)
        # g = nn.functional.sigmoid(t)
        # h = nn.functional.relu(self.hw_trans(chars_embeds))
        # chars_embeds = g * h + (1 - g) * chars_embeds

        embeds = self.word_embeds(sentence)
        


        # word_emebd_parameters = m.weight.numpy()
        if self.n_cap and self.cap_embedding_dim:
            cap_embedding = self.cap_embeds(caps)

        if self.n_cap and self.cap_embedding_dim:
            embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            embeds = torch.cat((embeds, chars_embeds), 1)

        if parameters['use_freq_vector']:
            frequency_embeddings = self.freq_embeds(sentence)
            embeds = torch.cat((embeds, frequency_embeddings), 1)
        if parameters['use_markdown_vector'] :
            markdown_embeddings = self.markdown_embeds(markdown)
            embeds = torch.cat((embeds, markdown_embeddings), 1)
        # print("embed before unsqueeze")
        # print(embeds.size())
        # print(embeds)
        embeds = embeds.unsqueeze(1) #what is done here
        # print("embed after unsqueeze")
        # print(embeds.size())
        # print(embeds)

        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)

        # print(lstm_out.size())
        # print(len(sentence))

        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0.0
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence_tokens, sentence, sentence_seg_preds,sentence_ctc_preds, tags, chars2, caps, chars2_length, d):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats = self._get_lstm_features_w_elmo(sentence_tokens, sentence, sentence_seg_preds, sentence_ctc_preds)

        # if self.use_elmo:
        #     if parameters['use_elmo_w_char']:
        #         feats = self._get_lstm_features_w_elmo_and_char(sentence_tokens, sentence, sentence_markdowns, chars2, caps, chars2_length, d)
        #     else:
        #         feats = self._get_lstm_features_w_elmo(sentence_tokens, sentence, sentence_seg_preds, sentence_ctc_preds)
        # else:
        #     feats = self._get_lstm_features(sentence,sentence_markdowns, chars2, caps, chars2_length, d)

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores


    def forward(self,sentence_tokens, sentence, sentence_seg_preds, sentence_ctc_preds, chars, caps, chars2_length, d):
        feats = self._get_lstm_features_w_elmo(sentence_tokens,  sentence, sentence_seg_preds,sentence_ctc_preds)

        # if self.use_elmo:
        #     if parameters['use_elmo_w_char']:
        #         feats = self._get_lstm_features_w_elmo_and_char(sentence_tokens, sentence,sentence_markdowns, chars, caps, chars2_length, d)
        #     else:
        #         feats = self._get_lstm_features_w_elmo(sentence_tokens,  sentence, sentence_markdowns,sentence_ctc_preds)
            
        # else:
        #     feats = self._get_lstm_features(sentence,sentence_markdowns, chars, caps, chars2_length, d)
        # viterbi to get tag_seq
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq
