# Running NER Model:

Downlaod all the pretrained in-domain word vectors and put them in the `resources/pretrained_word_vectors/`.

Inside the `NER` folder run the following command:

```
		python train_so.py 
```


By default it will show the evaluation on the `test` set. To evaluate on the dev set run the following command:

```
		python train_so.py -mode dev
```
