import numpy as np
import tensorflow as tf
from string import punctuation
import re
import os

import config


class TryLstm():
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.numDimensions = config.numDimensions
        self.maxSeqLength = config.maxSeqLength
        self.batchSize = config.batchSize
        self.lstmUnits = config.lstmUnits
        self.numClasses = config.numClasses
        self.load_gloves()

    def load_gloves(self):
        print("Loading gloves model")
        self.wordsList = np.load('model/wordsList.npy').tolist()
        self.wordsList = [word for word in self.wordsList]
        self.wordVectors = np.load('model/wordVectors.npy')

    def create_model(self):
        print("Creating LSTM model...")
        tf.reset_default_graph()

        labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        input_data = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength])

        data = tf.Variable(tf.zeros(
                        [self.batchSize, self.maxSeqLength, self.numDimensions]),
                        dtype=tf.float32
        )
        data = tf.nn.embedding_lookup(self.wordVectors, input_data)

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('models'))

        while True:
            original_tweet = input("Enter origin message: ")
            comment = input("Enter comment message: ")
            inputText = original_tweet + " < - > " + comment
            if original_tweet == "exit":
                break
                exit(1)
            inputMatrix = self.getSentenceMatrix(inputText)
            predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
            if (predictedSentiment[0] > predictedSentiment[1]):
                print("---Agreed Sentiment---")
                print('Agreed/Disagreed')
                print(f'{predictedSentiment[0]}/{predictedSentiment[1]}' + "\n")
            else:
                print("---Disagreed Sentiment---")
                print('Agreed/Disagreed')
                print(f'{predictedSentiment[0]}/{predictedSentiment[1]}' + "\n")
        exit(1)

    def clean_sentence(self, string):
        cleaned_string = ''
        for num, char in enumerate(string):
            if char == "<":
                if string[num + 2] == "-" and string[num + 4] == ">":
                    cleaned_string += char
            elif char == "-":
                if string[num - 2] == "<" and string[num + 2] == ">":
                    cleaned_string += char
            elif char == ">":
                if string[num - 4] == "<" and string[num - 2] == "-":
                    cleaned_string += char
            elif char not in punctuation:
                cleaned_string += char
        return cleaned_string

    def getSentenceMatrix(self, sentence):
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        sentenceMatrix = np.zeros([self.batchSize, self.maxSeqLength], dtype='int32')
        cleanedSentence = self.clean_sentence(sentence)
        split = cleanedSentence.split()
        for indexCounter, word in enumerate(split):
            try:
                sentenceMatrix[0, indexCounter] = self.wordsList.index(word)
            except ValueError:
                # Vector for unkown words
                sentenceMatrix[0, indexCounter] = 399999
        return sentenceMatrix


if __name__ == '__main__':
    var = TryLstm()
    var.create_model()
    exit(1)
