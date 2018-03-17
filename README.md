# LSTM neural network for sentence analysis.

A NN to predict the sentence's (dis)agreement.

## Prerequisites
To use this app you need:
```
* python3.6
* tensorflow
* numpy
```

## Installing
```
pip3.6 install -r req.txt
```

## Download Glove model:
You can download GloVe models from [stanford.edu](https://nlp.stanford.edu/projects/glove).

## Train data
Put train data into 'data/' folder with .polarity extension;

## Use

### To train model use:
```
python3.6 train_and_test.py --train='data/'
```

### To test pre-trained model:
```
python3.6 train_and_test.py --test='models/'
```
