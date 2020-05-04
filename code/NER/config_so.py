import optparse
import argparse
from collections import OrderedDict
import torch 
import utils_so as utils
import os


from datetime import date
today = str(date.today())


torch.backends.cudnn.deterministic = True




optparser = optparse.OptionParser()
parser = argparse.ArgumentParser()

parser.add_argument(
    "-T", "--train", default="../../resources/annotated_ner_data/StackOverflow/train.txt",
    help="Train set location"
)
parser.add_argument(
    "-t", "--test", default="../../resources/annotated_ner_data/StackOverflow/test.txt",
    help="Dev set location"
)
parser.add_argument(
    "-d", "--dev", default="../../resources/annotated_ner_data/StackOverflow/dev.txt",
    help="Test set location"
)
parser.add_argument(
    '--test_train', default="../../resources/annotated_ner_data/StackOverflow/train.txt",
    help='test train'
)

parser.add_argument(
    "-mode", "--mode", default="test",
    help="which file to evaluate, default test, set to dev if you want to evaluate on dev"
)

parser.add_argument(
    '--elmo_weight', default="../../resources/pretrained_word_vectors/ELMo/SO_elmo_weights.hdf5",
    help='elmo_weight file'
)

parser.add_argument(
    '--elmo_options', default="../../resources/pretrained_word_vectors/ELMo/SO_elmo_option.json",
    help='elmo_options file'
)
parser.add_argument(
    '--score', default='evaluation/temp/score.txt',
    help='score file location'
)
parser.add_argument(
    "-s", "--tag_scheme", default="iob",
    help="Tagging scheme (IOB or IOBES)"
)
parser.add_argument(
    "-l", "--lower", default="1",
    type=int, help="Lowercase words (this will not affect character inputs)"
)
parser.add_argument(
    "-z", "--zeros", default="0",
    type=int, help="Replace digits with 0"
)
parser.add_argument(
    "-c", "--char_dim", default="25",
    type=int, help="Char embedding dimension"
)
parser.add_argument(
    "-C", "--char_lstm_dim", default="25",
    type=int, help="Char LSTM hidden layer size"
)
parser.add_argument(
    "-b", "--char_bidirect", default="1",
    type=int, help="Use a bidirectional LSTM for chars"
)
parser.add_argument(
    "-w", "--word_dim", default="300",
    type=int, help="Token embedding dimension"
)
parser.add_argument(
    "-W", "--word_lstm_dim", default="300",
    type=int, help="Token LSTM hidden layer size"
)
parser.add_argument(
    "-B", "--word_bidirect", default="1",
    type=int, help="Use a bidirectional LSTM for words"
)
#JT
# parser.add_argument(
#     "-p", "--pre_emb", default="models/glove.6B.100d.txt",
#     help="Location of pretrained embeddings"
# )
parser.add_argument(
    "-p", "--pre_emb", default="../../resources/pretrained_word_vectors/Glove/vectors300.txt",
    help="Location of pretrained embeddings"
)
# parser.add_argument(
#     "-p", "--pre_emb", default="glove.6B.300d.txt",
#     help="Location of pretrained embeddings"
# )
parser.add_argument(
    "-all_emb", "--all_emb", default="0",
    type=int, help="Load all embeddings for dev and test"
)
parser.add_argument(
    "-cap_dim", "--cap_dim", default="0",
    type=int, help="Capitalization feature dimension (0 to disable)"
)
parser.add_argument(
    "-crf", "--crf", default="1",
    type=int, help="Use CRF (0 to disable)"
)
parser.add_argument(
    "-D", "--dropout", default=0.5,
    type=float, help="Droupout on the input (0 = no dropout)"
)
parser.add_argument(
    "-r", "--reload", default="0",
    type=int, help="Reload the last saved model"
)
parser.add_argument(
    "-use_gpu", '--use_gpu', default='0',
    type=int, help='whether or not to ues gpu'
)
parser.add_argument(
    "-gpu_id", '--gpu_id', default='0',
    type=int, help='which gpu to use'
)
parser.add_argument(
    '--loss', default='loss.txt',
    help='loss file location'
)
parser.add_argument(
    '--name', default='test',
    help='model name'
)
parser.add_argument(
    '--char_mode', choices=['CNN', 'LSTM'], default='CNN',
    help='char_CNN or char_LSTM'
)
#JT
parser.add_argument(
    '-use_pre_emb', '--use_pre_emb', default="0",
    type=int, help="Use pretrained emebdding (0 to disable)"
)
parser.add_argument(
    '-m', '--merge_tags', default="1",
    type=int, help="If we should merge tags (0 to disable)"
)
parser.add_argument(
    '--entity_category', choices=['code', 'human_lang', 'all'], default='all',
    help='which type of entity is intended to classify'
)
parser.add_argument(
    '-seed','--seed', default=9911, type=int, 
    help='which value to use for torch and numpy seed'
)
parser.add_argument(
    '-lr','--lr', default=0.015, type=float, 
    help='which learning rate to use'
)
parser.add_argument(
    '-epochs','--epochs', default=100, type=int, 
    help='number of epochs to train'
)
parser.add_argument(
    '-file_identifier','--file_identifier', default="", type=str, 
    help='file_identifier'
)


parser.add_argument(
    '-segmentation_only','--segmentation_only', default=0, type=int, 
    help='"If we should do segmentation only (1 to enable)'
)




parser.add_argument(
    "-use_elmo", '--use_elmo', default='1',
    type=int, help='whether or not to ues elmo'
)

parser.add_argument(
    "-use_elmo_w_char", '--use_elmo_w_char', default='0',
    type=int, help='whether or not to ues elmo with char embeds'
)



parser.add_argument(
    "-use_freq_vector", '--use_freq_vector', default='0',
    type=int, help='whether or not to ues the word frequency'
)
parser.add_argument(
    '--freq_vector_file', default="other_files/Freq_Vector.txt",
    help='elmo_options file'
)

parser.add_argument(
    '-freq_mapper_bin_count', '--freq_mapper_bin_count', default='100',
    type=int, help='how many bins to use in gaussian binning of the frequency vector'
)

parser.add_argument(
    '-freq_mapper_bin_width', '--freq_mapper_bin_width', default='5.0',
    type=float, help='the width of each bin for the gaussian binning of the frequency vector'
)



parser.add_argument(
    "-use_markdown_vector", '--use_markdown_vector', default='1',
    type=int, help='whether or not to ues the markdown from stackoverflow meta data'
)
parser.add_argument(
    "-use_segmentation_vector", '--use_segmentation_vector', default='1',
    type=int, help='whether or not to ues the code_pred vector'
)
parser.add_argument(
    "-use_han", '--use_han', default='1',
    type=int, help='whether or not to ues HAN networkd'
)


opts = parser.parse_args()
# print(args.char_mode)
# print(opts)

parameters = OrderedDict()

parameters['seed']=opts.seed



torch.manual_seed(parameters['seed'])


parameters["train_pred"]='auxilary_inputs_ner/segmenter_pred/segmenter_pred_train.txt'
parameters["dev_pred"]='auxilary_inputs_ner/segmenter_pred/segmenter_pred_dev.txt'
parameters["test_pred"]='auxilary_inputs_ner/segmenter_pred/segmenter_pred_test.txt'




parameters["ctc_pred"]='auxilary_inputs_ner/ctc_pred.tsv'


parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name
parameters['char_mode'] = opts.char_mode

parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()





use_gpu = parameters['use_gpu']
parameters["gpu_id"] = opts.gpu_id

parameters['use_pre_emb']=  opts.use_pre_emb == 1
parameters['merge_tags']=  opts.merge_tags == 1
parameters['entity_category'] = opts.entity_category
parameters['segmentation_only']=  opts.segmentation_only == 1


if opts.file_identifier=="":
    file_identifier = str(today)+"_"+str(parameters["seed"])+"_"
else:
    file_identifier=opts.file_identifier



parameters["models_path"] = "./models_"+file_identifier
parameters["eval_path"] = "./evaluation"
parameters["eval_temp"] = os.path.join(parameters["eval_path"], "temp_"+file_identifier)
parameters["eval_script"] = os.path.join(parameters["eval_path"], "conlleval")
parameters["perf_per_epoch_file"] = "perf_per_epoch_"+file_identifier+".txt"
parameters["sorted_entity_list_file_name"]="sorted_entity_list_by_count_all.json"



parameters["train"]=opts.train
parameters["dev"]=opts.dev
parameters["test"]=opts.test
parameters["LR"]=opts.lr
parameters["epochs"]=opts.epochs
parameters["mode"]=opts.mode



parameters['use_elmo'] = opts.use_elmo == 1 
parameters['use_elmo_w_char'] = opts.use_elmo_w_char == 1 
parameters["elmo_weight"]=opts.elmo_weight
parameters["elmo_options"]=opts.elmo_options
# parameters["elmo_layer"] = opts.elmo_layer
parameters["elmo_dim"]=1024


parameters['use_freq_vector'] = opts.use_freq_vector == 1 
parameters['freq_vector_file'] = opts.freq_vector_file
parameters["freq_mapper_bin_count"]=int(opts.freq_mapper_bin_count)
parameters["freq_mapper_bin_width"]=float(opts.freq_mapper_bin_width)
parameters['freq_dim']=parameters["freq_mapper_bin_count"]+2


parameters['use_markdown_vector'] = opts.use_markdown_vector == 1 
parameters['markdown_dim'] = 500
parameters['markdown_count'] = 3



parameters['use_segmentation_vector'] = True
parameters['segmentation_dim'] = 500
parameters['segmentation_count'] = 3


parameters['use_ner_pred_vector'] = True
parameters['ner_pred_dim'] = 500
parameters['ner_pred_count']=41


parameters['use_han'] = opts.use_han == 1 

parameters['use_code_recognizer_vector'] = True
parameters['code_recognizer_dim'] = 300
parameters['code_recognizer_count'] = 3


parameters['embedding_context_vecotr_size'] = 300
parameters['word_context_vecotr_size'] = 300


# parameters['markdown_dim']=parameters['elmo_dim']
# parameters['segmentation_dim']=parameters['elmo_dim']
# parameters['code_recognizer_dim']=parameters['elmo_dim']
# parameters['ner_pred_dim']=parameters['elmo_dim']



parameters["vocab_count_file"]="train_vocab_w_count.json"


parameters["entity_category_code"]=["Class", "Library_Class", "Class_Name",
                        "Function", "Library_Function","Function_Name",
                        "Variable_Name", "Library_Variable","Variable",
                        "Library",
                        "Code_Block",
                        "Version", "File_Name",
                        "Output_Block"]
parameters["entity_category_human_language"]=[ "Data_Structure", "Data_Type", "Algorithm",
                            "User_Interface_Element", "Device", 
                            "Website","Organization","Website_Organization",
                            "Application", "Language",   "File_Type", "Operating_System", "HTML_XML_Tag",
                            "Error_Name","Keyboard_IP", "User_Name",
                            "Output_Block"]

