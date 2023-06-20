# MusicVAE
Implementation of [MusicVAE](http://proceedings.mlr.press/v80/roberts18a/roberts18a.pdf) in PyTorch 


## Dataset
1. Please download Groove MIDI dataset from this [link](https://magenta.tensorflow.org/datasets/groove) and put the dataset in the root directory.
2. Divide the dataset into three sets: training, validation, and testing. Additionally, preprocess the data into the tfrecord format. To accomplish this, you can execute the provided command.
```
python preprocess.py

```

##