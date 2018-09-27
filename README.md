# Emergence of Communication in an Interactive World with Consistent Speakers

Author implementation of the paper: https://arxiv.org/pdf/1809.00549.pdf


#### Pretraining

```
$ python run.py pretrain_speaker
$ python run.py pretrain_listener
```

#### Training

```
$ python run.py train_joint 
     --restore-speaker=experiments/pretrain_speaker/path.to.ckp
     --restore-listener=experiments/pretrain_listener/path.to.ckp
```
