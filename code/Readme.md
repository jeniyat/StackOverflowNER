# Running BERT NER Model:

Download the [bert-word-piece-softner.zip](https://mega.nz/file/aBgWESJB#0LDKqizbTAlWHUReUSiJ-prgc2LngjzmktUzMC3-Jk0) and [word_piece_1_0m.zip](https://mega.nz/file/yE5WACBI#v1CQLM7I9451NVF2SBG8zu3UQvWAml7DAk6tybh9HkA) and unzip inside the `BERT_NER/fine-tune/` folder.

To extract the predictions on dev and test set, first install the transformer from by running `pip install .` inside the `BERT_NER/transformers/` folder. Then run the following command inside the `BERT_NER/fine-tune/` folder:

```
    bash run_predict.sh
```

- It will print the perfromance of the model on dev and test set at the `stdout` 
- It will save the predictions on dev and test set at `bert-word-piece-softner/dev_predictions.txt`, `bert-word-piece-softner/test_predictions.txt` respectively.

To train the model run the following command inside the `BERT_NER/fine-tune/` folder:

```
    bash run_train.sh
```

This BERT-NER uses the pretrained stackoverflow domain bert-representations from [BERTOverflow](https://github.com/lanwuwei/BERTOverflow).

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

