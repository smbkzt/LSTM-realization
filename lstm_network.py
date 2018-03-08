import os
import pickle
import datetime
from os import listdir
from random import randint
from string import punctuation
from os.path import isfile, join

import numpy as np
import tensorflow as tf


class PrepareData():
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.maxSeqLength = 30
        self.current_state = 0
        self.line_number = 0
        self.load_glove_model()
        self.calculate_lines('data/agreed.polarity', 'data/disagreed.polarity')
        # self.create_idx()

    def clean_string(self, string):
        split_ = string.split()
        cleaned_string = ""
        for char in string:
            if char not in punctuation:
                cleaned_string += char
        return cleaned_string

    def load_glove_model(self):
        self.wordsList = np.load('model/wordsList.npy')
        self.wordsList = self.wordsList.tolist()
        self.wordVectors = np.load('model/wordVectors.npy')

    def calculate_lines(self, agreed, dis):
        for file in [agreed, dis]:
            num_of_lines = 0
            with open(file, 'r') as f:
                for line in f.readlines():
                    num_of_lines += 1
            self.line_number += num_of_lines

    def create_idx(self):
        ids = np.zeros((self.line_number, self.maxSeqLength), dtype='int32')
        filesList = [f for f in listdir('data/')
                     if isfile(join('data/', f)) and f.endswith(".polarity")]
        fileCounter = 0
        hm_lines = 11000
        for file in sorted(filesList):
            with open(f"data/{file}", "r", errors='ignore') as f:
                lines = f.readlines()[:hm_lines]
                for num, line in enumerate(lines):
                    if num % 1000 == 0:
                        print(num + self.current_state)
                    cleanedLine = self.clean_string(line)
                    split = cleanedLine.split()
                    for w_num, word in enumerate(split):
                        try:
                            getWordIndex = self.wordsList.index(word)
                            ids[self.current_state + num][w_num] = getWordIndex
                        except ValueError:
                            ids[self.current_state + num][w_num] = 000
                        if w_num >= self.maxSeqLength - 1:
                            break
            self.current_state += hm_lines
        np.save('model/idsMatrix', ids)
        print("Saved ids matrix to the 'model/idsMatrix';")


class RNNModel(PrepareData):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.ids = np.load('model/idsMatrix.npy')
        self.batchSize = 24
        self.lstmUnits = 64
        self.numClasses = 2
        self.iterations = 11000
        self.numDimensions = 50
        self.create_model()

    def get_train_batch(self):
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        for i in range(self.batchSize):
            if (i % 2 == 0):
                num = randint(1, 10000)
                labels.append([1, 0])  # Agreed
            else:
                num = randint(12000, 22000)
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num-1:num]
        return arr, labels

    def get_test_batch(self):
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        for i in range(self.batchSize):
            num = randint(10000, 12000)
            if (num <= 11000):
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            arr[i] = self.ids[num-1:num]
        return arr, labels

    def test_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint('models'))
        iterations = 100
        accuracy = 0
        for i in range(iterations):
            nextBatch, nextBatchLabels = self.get_test_batch()
            current_acc = self.sess.run(self.accuracy, {self.input_data: nextBatch, self.labels: nextBatchLabels}) * 100
            accuracy += current_acc
            if i > 0:
                av_acc = accuracy / i
                print("Average accuracy: ", av_acc)

    def create_model(self):
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        self.input_data = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength])

        data = tf.Variable(tf.zeros([self.batchSize,
                                     self.maxSeqLength,
                                     self.numDimensions]), dtype=tf.float32)

        data = tf.nn.embedding_lookup(self.wordVectors, self.input_data)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(
            cell=lstmCell,
            output_keep_prob=0.75
        )
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal(
            [self.lstmUnits,
             self.numClasses])
        )
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)
        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                              logits=prediction, labels=self.labels)
                              )
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()
        logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        self.writer = tf.summary.FileWriter(logdir, self.sess.graph)

    def train_model(self):
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.iterations):
            print("Iterations: ", i)
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.get_train_batch()
            self.sess.run(self.optimizer, {self.input_data: nextBatch,
                                           self.labels: nextBatchLabels}
                          )
            # Write summary to Tensorboard
            if (i % 50 == 0):
                summary = self.sess.run(self.merged, {self.input_data: nextBatch, self.labels: nextBatchLabels})
                self.writer.add_summary(summary, i)

            # Save the network every 10,000 training iterations
            if (i % 1000 == 0 and i != 0):
                save_path = saver.save(self.sess, "models/pretrained_lstm.ckpt", global_step=i)
                print("Saved to %s" % save_path)
        self.writer.close()


if __name__ == '__main__':
    prep = RNNModel()
    prep.train_model()
    prep.test_model()
