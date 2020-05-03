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

By default this code base will run on GPU. You can disable it by the following:

```
		python train_so.py -use_gpu 0
```

By default this code base will run on GPU ID `0`. You can change the gpu id by the following:

```
		python train_so.py -gpu_id <gpu_id>
		python train_so.py -gpu_id 1
```