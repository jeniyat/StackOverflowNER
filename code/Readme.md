# Running BERT NER Model:

## Prerequisitie:
1. Install the modified version of [huggingface transformers](https://github.com/lanwuwei/BERTOverflow). This respository contains all the necessary modification we made in the to adapt the `BertTokenClassifier` with embedding level attention.

2. Download the [utils_fine_tune.zip](https://mega.nz/file/bRp3lTBY#lHamCVxeVr6wfsdjFpgqimvWZ5vJoeRyoaU40-7pl5c) and unzip inside `BERT_NER`.

3. Download the [data_ctc.zip](https://mega.nz/file/DVYUkATS#DqDKlYPT2zfXSaAy5oTvolNrBLJzS5bRV5m_m3qUreU) and unzip. Update the `parameters_ctc['RESOURCES_base_directory']` path with the abosolute path of the unzipped folder. The `parameters_ctc['RESOURCES_base_directory']` is defined inside the `utils_ctc/config_ctc.py` file.


## Extract the prediction on a new input file:

To extract the software entities from a given file run the following:

```
    python E2E_SoftNER.py --input_file_with_so_body xml_filted_body
```

- It will save the predictions on the input file at `ner_preds.txt`


## Replicate the reported results:



# Running Attentive-BiLSTM NER Model:

Downlaod all the pretrained in-domain word vectors and put them in the `resources/pretrained_word_vectors/`.

Inside the `NER` folder run the following command:

```
    python train_so.py 
```


By default it will show the evaluation on the `test` set. To evaluate on the dev set run the following command:

```
    python train_so.py -mode dev
```

By default this code base will run on GPU. You can disable it by the `-use_gpu` paramerter as below:

```
    python train_so.py -use_gpu 0
```

By default this code base will run on GPU ID `0`. You can change the gpu id by the `-gpu_id` paramerter as below:

```
    python train_so.py -gpu_id 1
```


# Loading the annotated files:

To read the dataset only use the loader_so.py file from `DataReader` folder as below:


```
    import loader_so
    path_to_file = "../../resources/annotated_ner_data/StackOverflow/train.txt"
    all_sentneces = loader_so.loader_so_text(path_to_file)
 
```

By default the `loader_so_text` function merges the following 6 entities to 3 as below: 

```
    "Library_Function" -> "Function"
    "Function_Name" -> "Function"

    "Class_Name" -> "Class"
    "Library_Class" -> "Class"

    "Library_Variable" -> "Variable"
    "Variable_Name" -> "Variable"

    "Website" -> "Website"
    "Organization" -> "Website"

```

To skip this merging, set `merge_tag= False` as below:

```
    import loader_so
    path_to_file = "../../resources/annotated_ner_data/StackOverflow/train.txt"
    all_sentneces = loader_so.loader_so_text(path_to_file,merge_tag=False)
 
```


By default the `loader_so_text` function will convert the 5 low frequency enttiy as "O". To skip this conversion, set `replace_low_freq_tags= False` as below:



```
    import loader_so
    path_to_file = "../../resources/annotated_ner_data/StackOverflow/train.txt"
    all_sentneces = loader_so.loader_so_text(path_to_file, replace_low_freq_tags= False)
 
```

# Run the Tokenizer:

To tokenized the code-mixed texts from StackOverflow utilized the source codes insides the `SOTokenizer` folder as below:

```
	import stokenizer
	sentence = 'I do think that the request I send to my API should be more like {post=>{"kind"=>"GGG"}} and not {"kind"=>"GGG"}.'
	tokens = stokenizer.tokenize(sentence)
	print("tokens: ",tokens)

```
Tokenized Output:

```
	tokens:  ['I', 'do', 'think', 'that', 'the', 'request', 'I', 'send', 'to', 'my', 'API', 'should', 'be', 'more', 'like', ' { post=> { "kind"=>"GGG" }  } ', 'and', 'not', ' { "kind"=>"GGG" } ', '.']

```
# Resource

All the files required can be found here: https://mega.nz/folder/ycxnmSJL#8ZQgHEqBAaGbij3uAHhsSw
