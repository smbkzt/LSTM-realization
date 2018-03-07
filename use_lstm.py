import numpy as np
import tensorflow as tf
from string import punctuation
import re
import os


class TryLstm():
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.numDimensions = 50
        self.maxSeqLength = 30
        self.batchSize = 24
        self.lstmUnits = 64
        self.numClasses = 2
        self.load_gloves()

    def load_gloves(self):
        self.wordsList = np.load('model/wordsList.npy').tolist()
        # Encode words as UTF-8
        self.wordsList = [word for word in self.wordsList]
        self.wordVectors = np.load('model/wordVectors.npy')

    def create_model(self):
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
            inputText = input("Enter the text or 'exit' to quit: ")
            if inputText == "exit":
                break
            inputMatrix = self.getSentenceMatrix(inputText)
            predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
            if (predictedSentiment[0] > predictedSentiment[1]):
                print("---Positive Sentiment---")
                print('predictedSentiment/Negative')
                print(f'{predictedSentiment[0]}/{predictedSentiment[1]}' + "\n")
            else:
                print("---Negative Sentiment---")
                print('predictedSentiment/Negative\n')
                print(f'{predictedSentiment[0]}/{predictedSentiment[1]}' + "\n")
        exit(1)

    def cleanSentences(self, string):
        clean = ''
        for char in string:
            if char in punctuation:
                continue
            clean += char
        return clean

    def getSentenceMatrix(self, sentence):
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        sentenceMatrix = np.zeros([self.batchSize, self.maxSeqLength], dtype='int32')
        cleanedSentence = self.cleanSentences(sentence)
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
