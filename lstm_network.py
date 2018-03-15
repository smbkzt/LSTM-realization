import os
import pickle
import datetime
from os import listdir
from random import randint
from string import punctuation
from os.path import isfile, join

import numpy as np
import tensorflow as tf

import config


class PrepareData():
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.maxSeqLength = config.maxSeqLength
        self.current_state = 0
        self.load_glove_model()
        self.calculate_lines('data/agreed.polarity', 'data/disagreed.polarity')
        print("Agreement examples ", self.agr_lines)
        print("Disagreement examples ", self.dis_lines)
        self.line_number = self.agr_lines + self.dis_lines
        # self.create_idx()

    def clean_string(self, string):
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

    def load_glove_model(self):
        self.wordsList = np.load('model/wordsList.npy')
        self.wordsList = self.wordsList.tolist()
        self.wordVectors = np.load('model/wordVectors.npy')

    def calculate_lines(self, agreed, dis):
        for file in [agreed, dis]:
            with open(file, 'r') as f:
                lines = f.readlines()
            if "data/agreed.polarity" == file:
                self.agr_lines = len(lines)
            else:
                self.dis_lines = len(lines)

    def create_idx(self):
        ids = np.zeros((self.line_number, self.maxSeqLength), dtype='int32')
        filesList = [f for f in listdir('data/')
                     if isfile(join('data/', f)) and f.endswith(".polarity")]
        fileCounter = 0
        # hm_lines = 299
        for file in sorted(filesList):
            with open(f"data/{file}", "r", errors='ignore') as f:
                print(f"Started readinf file - {file}....")
                lines = f.readlines()
                for num, line in enumerate(lines):
                    if num % 100 == 0:
                        current_line = num + self.current_state
                        print(f"Reading line number: {current_line} / {self.line_number}")
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
            self.current_state += len(lines)
        np.save('model/idsMatrix', ids)
        print("Saved ids matrix to the 'model/idsMatrix';")


class RNNModel(PrepareData):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.ids = np.load('model/idsMatrix.npy')
        self.batchSize = config.batchSize
        self.lstmUnits = config.lstmUnits
        self.numClasses = config.numClasses
        self.iterations = 1001
        self.numDimensions = config.numDimensions
        self.create_model()

    def get_train_batch(self):
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        for i in range(self.batchSize):
            if (i % 2 == 0):
                num = randint(1, int(self.agr_lines-(self.agr_lines*0.1)))
                labels.append([1, 0])  # Agreed
            else:
                num = randint(int(self.agr_lines + (self.dis_lines*0.1)), int(self.agr_lines + self.dis_lines))
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num-1:num]
        return arr, labels

    def get_test_batch(self):
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        for i in range(self.batchSize):
            num = randint(int(self.agr_lines-(self.agr_lines*0.1)), int(self.agr_lines + (self.dis_lines*0.1)))
            if (num <= self.agr_lines):
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
        print("Creating training model...")
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
            if (i % 100 == 0):
                summary = self.sess.run(self.merged, {self.input_data: nextBatch, self.labels: nextBatchLabels})
                self.writer.add_summary(summary, i)

            # Save the network every 10,000 training iterations
            if (i % 100 == 0 and i != 0):
                save_path = saver.save(self.sess, "models/pretrained_lstm.ckpt", global_step=i)
                print("Saved to %s" % save_path)
        self.writer.close()


if __name__ == '__main__':
    prep = RNNModel()
    prep.train_model()
    prep.test_model()
