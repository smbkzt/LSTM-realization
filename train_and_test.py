import os
import pickle
import datetime
import argparse
from os import listdir
from random import randint
from string import punctuation
from os.path import isfile, join

import numpy as np
import tensorflow as tf

import config


class PrepareData():
    """Preparing dataset to be inputed in TF"""
    def __init__(self, path):
        self.dataset_path = path
        self.maxSeqLength = config.maxSeqLength
        self.current_state = 0
        self.load_glove_model()
        self.calculate_lines()
        self.check_idx_matrix()

    def clean_string(self, string) -> str:
        """Cleans messages from punctuation"""
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
        """Loads the glove model"""
        self.wordsList = np.load('data/wordsList.npy')
        self.wordsList = self.wordsList.tolist()
        self.wordVectors = np.load('data/wordVectors.npy')

    def calculate_lines(self) -> str:
        # Get the list of all files in folder
        self.filesList = [self.dataset_path + f for f in listdir(self.dataset_path)
                          if isfile(join(self.dataset_path, f)) and
                          f.endswith(".polarity")]
        for file in self.filesList:
            with open(file, 'r') as f:
                lines = f.readlines()
            if "data/agreed.polarity" == file:
                self.agr_lines = len(lines)
            else:
                self.dis_lines = len(lines)
        print("Agreement examples ", self.agr_lines)
        print("Disagreement examples ", self.dis_lines)
        self.line_number = self.agr_lines + self.dis_lines

    def check_idx_matrix(self):
        """Checks if any idx matrix exists"""
        idsMatrix = [self.dataset_path + f for f in listdir(self.dataset_path)
                     if isfile(join(self.dataset_path, f)) and
                     f.endswith("idsMatrix.npy")]
        if len(idsMatrix) >= 1:
            ans = input("Found 'idsMatrix'. Would you like to recreate it?y/n ")
            if ans in ["y", "", "Yes", "Y"]:
                self.create_idx()
            else:
                print("Continue...")
        else:
            print("Haven't found the idx matrix models.")
            self.create_idx()

    def create_idx(self):
        """Function of idx creation"""
        ids = np.zeros((self.line_number, self.maxSeqLength),
                       dtype='int32')
        for file in sorted(self.filesList):
            with open(f"{file}", "r", errors='ignore') as f:
                print(f"\nStarted reading file - {file}....")
                lines = f.readlines()
                for num, line in enumerate(lines):
                    if num % 100 == 0:
                        current_line = num + self.current_state
                        print(f"Reading line number: {current_line}/{self.line_number}")
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
            # To continue from "checkpoint"
            self.current_state += len(lines)
        np.save('data/idsMatrix', ids)
        print("Saved ids matrix to the 'model/idsMatrix';")


class RNNModel(PrepareData):
    """Class of TF models creation"""
    def __init__(self, path="uknown"):
        dir_path = path
        if not dir_path == "uknown":
            super(RNNModel, self).__init__(dir_path)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Avoid the tf warnings
        self.batchSize = config.batchSize
        self.lstmUnits = config.lstmUnits
        self.numClasses = config.numClasses
        self.numDimensions = config.numDimensions
        self.maxSeqLength = config.maxSeqLength

    def get_train_batch(self):
        '''Returning training batch function'''
        self.ids = np.load('data/idsMatrix.npy')
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        for i in range(self.batchSize):
            if (i % 2 == 0):
                num = randint(1, int(self.agr_lines-(self.agr_lines*0.1)))
                labels.append([1, 0])  # Agreed
            else:
                num = randint(int(self.agr_lines + (self.dis_lines*0.1)),
                              int(self.agr_lines + self.dis_lines))
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num-1:num]
        return arr, labels

    def get_test_batch(self):
        '''Returning training batch function'''
        self.ids = np.load('data/idsMatrix.npy')
        with open("data/agreed.polarity") as f:
            agr_lines = len(f.readlines())
        with open("data/disagreed.polarity") as f:
            dis_lines = len(f.readlines())
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        from_line = int(agr_lines-(agr_lines*0.1))
        to_line = int(agr_lines + (dis_lines*0.1))
        for i in range(self.batchSize):
            num = randint(from_line, to_line)
            if (num <= agr_lines):
                labels.append([1, 0])  # Agreed
            else:
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num-1:num]
        return arr, labels

    def create_and_train_model(self):
        """Creates the TF model"""
        print("Creating training model...")
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        labels = tf.placeholder(tf.float32,
                                [self.batchSize, self.numClasses])
        tf.add_to_collection("labels", labels)

        input_data = tf.placeholder(tf.int32,
                                    [self.batchSize, self.maxSeqLength])
        # We are saving to the collections, in order to resore it later
        tf.add_to_collection("input_data", input_data)

        data = tf.Variable(tf.zeros([self.batchSize,
                                     self.maxSeqLength,
                                     self.numDimensions]), dtype=tf.float32)

        data = tf.nn.embedding_lookup(self.wordVectors, input_data)
        cells = []
        for _ in range(config.cells):
            lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
            lstmCell = tf.contrib.rnn.DropoutWrapper(
                cell=lstmCell,
                output_keep_prob=0.75
            )
            cells.append(lstmCell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        value, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal(
            [self.lstmUnits,
             self.numClasses])
        )
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)

        prediction = (tf.matmul(last, weight) + bias)
        # Here we are doing the same
        tf.add_to_collection("prediction", prediction)

        correctPred = tf.equal(tf.argmax(prediction, 1),
                               tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        tf.add_to_collection("accuracy", accuracy)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                              logits=prediction, labels=labels)
                              )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        merged = tf.summary.merge_all()

        # ------ Below is training process ---------
        logdir = "tensorboard/" + \
                 datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(config.training_steps):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.get_train_batch()
            sess.run(optimizer, {input_data: nextBatch,
                                 labels: nextBatchLabels}
                     )
            # Write summary to Tensorboard
            if (i % 100 == 0):
                print("Iterations: ", i)
                summary = sess.run(merged,
                                   {input_data: nextBatch,
                                    labels: nextBatchLabels}
                                   )
                writer.add_summary(summary, i)
            # Save the network every 10,000 training iterations
            # if (i % 1000 == 0 and i != 0):
            #     save_path = saver.save(sess,
            #                            "models/pretrained_lstm.ckpt",
            #                            global_step=i)
            #     print(f"Saved to {save_path}")

        save_path = saver.save(sess, "models/pretrained_lstm.ckpt",
                               global_step=config.training_steps)
        writer.close()
        sess.close()

    def test_model(self, dir_):
        # Starting the session
        with tf.Session() as sess:
            path = ".".join([tf.train.latest_checkpoint(dir_), "meta"])
            # Get collections
            saver = tf.train.import_meta_graph(path)
            accuracy = tf.get_collection("accuracy")[0]
            input_data = tf.get_collection("input_data")[0]
            labels = tf.get_collection("labels")[0]

            saver.restore(sess, tf.train.latest_checkpoint(dir_))
            print("Testing pre-trained model....")

            iterations = 100
            accuracy_int = 0
            av_acc = 1
            for i in range(iterations):
                nextBatch, nextBatchLabels = self.get_test_batch()
                cur_acc = sess.run(accuracy,
                                   {input_data: nextBatch,
                                    labels: nextBatchLabels}
                                   ) * 100
                if i > 0:
                    accuracy_int += cur_acc
                    av_acc = accuracy_int / i
                    print("Test batch #", i,
                          f"Curr/Avg accuracy:{int(cur_acc)}/{int(av_acc)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model")
    parser.add_argument("--test", help="Test trained model")
    args = parser.parse_args()
    if args.train:
        train = RNNModel(args.train)
        train.create_and_train_model()
    elif args.test:
        test = RNNModel()
        test.test_model(args.test)
