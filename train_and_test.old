import os
import re
import datetime
import argparse
import requests
# from lxml.html import fromstring

from os import listdir
from random import randint
from string import punctuation
from os.path import isfile, join

import numpy as np
import tensorflow as tf

import config

requests.packages.urllib3.disable_warnings()


class PrepareData():
    """Preparing dataset to be inputed in TF"""

    def __init__(self, path: str):
        self.dataset_path = path
        self.maxSeqLength = config.maxSeqLength
        self.current_state = 0
        self.check_idx_matrix_occurance()

    @staticmethod
    def clean_string(string: str) -> str:
        """Cleans messages from punctuation and mentions"""
        string = re.sub(r"@[A-Za-z0-9]+", "", string)  # Delete tweet mentions
        # tweets = string.split(" < - > ")
        # for tweet in tweets:
        #     if len(tweet) < 50:
        #         url = re.search('https?://[A-Za-z0-9./]+', tweet)
        #         if url:
        #             try:
        #                 reponse = requests.get(url.group(0), verify=False)
        #                 tree = fromstring(reponse.content)
        #                 title = tree.findtext('.//title')
        #                 print(string)
        #                 print(title + "\n")
        #                 string = re.sub('https?://[A-Za-z0-9./]+',
        #                                 f' {title} ',
        #                                 string)
        #             except Exception as error:
        #                 print(error)
        #     else:
        #         string = re.sub('https?://[A-Za-z0-9./]+', '', string)
        string = re.sub('https?://[A-Za-z0-9./]+', '', string.lower())
        cleaned_string = ''
        string = string.split(" < - > ")
        for num, part in enumerate(string, 1):
            for char in part:
                if char not in punctuation:
                    cleaned_string += char
            if num == 1:
                cleaned_string += " < - > "
        repeated_found = re.search(r'\s{2,}', cleaned_string)
        if repeated_found:
            cleaned_string = re.sub(r'\s{2,}', " ", cleaned_string)
        return cleaned_string

    def load_glove_model(self):
        """Loads the glove model"""
        self.wordsList = np.load('data/wordsList.npy')
        self.wordsList = self.wordsList.tolist()

    def get_files_list(self, path,  endswith):
        list_of_files = [path + f for f
                         in listdir(path)
                         if isfile(join(path, f)) and
                         f.endswith(endswith)]
        return list_of_files

    def calculate_lines(self) -> int:
        # Get the list of all files in folder
        self.filesList = self.get_files_list(self.dataset_path, ".polarity")

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

    def check_idx_matrix_occurance(self):
        """Checks if any idx matrix exists"""
        self.calculate_lines()
        rnn = RNNModel()
        rnn.agr_lines = self.agr_lines
        rnn.dis_lines = self.dis_lines
        idsMatrix = self.get_files_list(self.dataset_path, "idsMatrix.npy")
        if len(idsMatrix) >= 1:
            ans = input(
                "Found 'idsMatrix'. Would you like to recreate it? (y/n) ")
            if ans in ["y", "", "Yes", "Y"]:
                self.create_idx()
            else:
                print("Continue...")
        else:
            print("Haven't found the idx matrix models.")
            self.create_idx()
        rnn.create_and_train_model()

    def create_idx(self):
        """Function of idx creation"""
        self.load_glove_model()
        ids = np.zeros((self.line_number, self.maxSeqLength),
                       dtype='int32')
        for file in sorted(self.filesList):
            with open(f"{file}", "r", errors='ignore') as f:
                print(f"\nStarted reading file - {file}....")
                lines = f.readlines()
                for num, line in enumerate(lines):
                    if num % 100 == 0:
                        current_line = num + self.current_state
                        print(
                            f"Reading line number: \
                            {current_line}/{self.line_number}")
                    cleaned_line = self.clean_string(line)
                    split = cleaned_line.split()
                    for w_num, word in enumerate(split):
                        try:
                            get_word_index = self.wordsList.index(word)
                            ids[self.current_state + num][w_num] = \
                                get_word_index
                        except ValueError:
                            # repeated_found = re.match(r'(.)\1{2,}', word)
                            # if repeated_found:
                            #     print(word)
                            ids[self.current_state + num][w_num] = 399999
                        if w_num >= self.maxSeqLength - 1:
                            break
            # To continue from "checkpoint"
            self.current_state += len(lines)
        np.save('data/idsMatrix', ids)
        print("Saved ids matrix to the 'model/idsMatrix';")


class RNNModel():
    """Class of TF models creation"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Avoid the tf warnings

    def __init__(self):
        self.batchSize = config.batchSize
        self.lstmUnits = config.lstmUnits
        self.numClasses = config.numClasses
        self.numDimensions = config.numDimensions
        self.maxSeqLength = config.maxSeqLength
        self.wordVectors = np.load('data/wordVectors.npy')

    def get_train_batch(self):
        """Returning training batch function"""
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        for i in range(self.batchSize):
            if i % 2 == 0:
                num = randint(1, int(self.agr_lines - (self.agr_lines * 0.1)))
                labels.append([1, 0])  # Agreed
            else:
                num = randint(int(self.agr_lines + (self.dis_lines * 0.1)),
                              int(self.agr_lines + self.dis_lines))
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num - 1:num]
        return arr, labels

    def get_test_batch(self):
        """Returning training batch function"""
        with open("data/agreed.polarity") as f:
            agr_lines = len(f.readlines())
        with open("data/disagreed.polarity") as f:
            dis_lines = len(f.readlines())
        labels = []
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        from_line = int(agr_lines - (agr_lines * 0.1))
        to_line = int(agr_lines + (dis_lines * 0.1))
        for i in range(self.batchSize):
            num = randint(from_line, to_line)
            if num <= agr_lines:
                labels.append([1, 0])  # Agreed
            else:
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num - 1:num]
        return arr, labels

    def create_and_train_model(self):
        """Creates the TF model"""
        self.ids = np.load('data/idsMatrix.npy')
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
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                cell=lstm_cell,
                output_keep_prob=0.75
            )
            cells.append(lstm_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        initial_state = cell.zero_state(self.batchSize, tf.float32)
        value, final_state = tf.nn.dynamic_rnn(cell, data,
                                               initial_state=initial_state,
                                               dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal(
            [self.lstmUnits,
             self.numClasses])
        )
        bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        # Adding prediction to histogram
        tf.summary.histogram('predictions', prediction)

        # Here we are doing the same
        tf.add_to_collection("prediction", prediction)

        correct_pred = tf.equal(tf.argmax(prediction, 1),
                                tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.add_to_collection("accuracy", accuracy)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=prediction, labels=labels)
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        tf.add_to_collection("optimizer", optimizer)
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.histogram("Out", value[:, -1])
        merged = tf.summary.merge_all()

        # ------ Below is training process ---------
        log_dir = "tensorboard/" + \
                  datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(config.training_steps+1):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.get_train_batch()
            sess.run(optimizer, {input_data: nextBatch,
                                 labels: nextBatchLabels}
                     )
            # Write summary to Tensorboard
            if i % 100 == 0:
                print(f"Iterations: {i}/{config.training_steps}")
                summary = sess.run(merged,
                                   {input_data: nextBatch,
                                    labels: nextBatchLabels}
                                   )
                writer.add_summary(summary, i)
            if i % 200 == 0 and i != 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(
                    self.batchSize, tf.float32))
                nextBatch, nextBatchLabels = self.get_test_batch()
                feed = {input_data: nextBatch,
                        labels: nextBatchLabels,
                        initial_state: val_state}
                summary, batch_acc, val_state = sess.run(
                    [merged, accuracy, final_state], feed_dict=feed)
                val_acc.append(batch_acc)
                print("\nVal acc: {:.3f}\n".format(np.mean(val_acc)))
            # Save the network every 10,000 training iterations
            # if (i % 1000 == 0 and i != 0):
            #     save_path = saver.save(sess,
            #                            "models/pretrained_lstm.ckpt",
            #                            global_step=i)
            #     print(f"Saved to {save_path}")

        path = saver.save(sess, "models/pretrained_lstm.ckpt",
                          global_step=config.training_steps)
        print(f"Model saved to: {path}")
        writer.close()
        sess.close()

    def test_model(self, dir_):
        # Starting the session
        self.ids = np.load('data/idsMatrix.npy')
        with tf.Session() as sess:
            path = ".".join([tf.train.latest_checkpoint(dir_), "meta"])
            # Get collections
            saver = tf.train.import_meta_graph(path)
            accuracy = tf.get_collection("accuracy")[0]
            input_data = tf.get_collection("input_data")[0]
            labels = tf.get_collection("labels")[0]

            saver.restore(sess, tf.train.latest_checkpoint(dir_))
            print("Testing pre-trained model....")
            test_acc = []
            for i in range(20):
                nextBatch, nextBatchLabels = self.get_test_batch()
                cur_acc = sess.run(accuracy,
                                   {input_data: nextBatch,
                                    labels: nextBatchLabels}
                                   )
                test_acc.append(cur_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model")
    parser.add_argument("--test", help="Test trained model")
    args = parser.parse_args()
    if args.train:
        train = PrepareData(args.train)
    elif args.test:
        test = RNNModel()
        test.test_model(args.test)
