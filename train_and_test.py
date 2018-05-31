import os
import re
import datetime
import argparse
import pickle
# import requests
# from lxml.html import fromstring

from os import listdir
from random import randint
# from string import punctuation
from os.path import isfile, join

import numpy as np
import tensorflow as tf

import config

# requests.packages.urllib3.disable_warnings()


class PrepareData():
    """Preparing dataset to be inputed in TF"""

    def __init__(self, path: str):
        self.__dataset_path = path if path.endswith("/") else path + "/"
        self.__maxSeqLength = config.maxSeqLength
        self.__overall_line_number = 0
        self.__check_idx_matrix_occurance()

    @staticmethod
    def clean_string(string: str) -> str:
        """Cleans messages from punctuation and mentions"""

        ascii_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '-1234567890"
        seperator = " < - > "
        cleaned_string = ''
        cut_sentence_until = int(config.maxSeqLength/2) - int(len(seperator)/2)

        #  Delete tweet mentions
        string = re.sub(r"@[A-Za-z0-9]+", "", string)

        # Replace urls with website titles
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

        # Delete the urls
        string = re.sub('https?://[A-Za-z0-9./]+', '', string.lower())

        # Delete all punctuation marks
        string = string.split(seperator)
        for num, part in enumerate(string, 1):
            for char in part:
                if char in ascii_letters:
                    cleaned_string += char
            if num == 1:
                cleaned_string += seperator

        # delete repeated whitespaces (more than 2)
        if re.search(r'\s{2,}', cleaned_string):
            cleaned_string = re.sub(r'\s{2,}', " ", cleaned_string)

        # Check whether the length of the sentences are more than max+50
        # If it is max, cut 2 sentences from the center (seperator)
        if len(cleaned_string.split()) > config.maxSeqLength + 50:
            new_line = " "
            cleaned_string = cleaned_string.split(seperator)
            for number, line in enumerate(cleaned_string):
                line_ = line.split(" ")[:cut_sentence_until]
                for word in line_:
                    new_line += word
                    new_line += " "
                if number == 0:
                    new_line += seperator
            cleaned_string = new_line
        return cleaned_string

    def __get_words_list(self) -> list:
        """Loads the glove model"""

        wordsList = np.load('data/wordsList.npy')
        wordsList = wordsList.tolist()
        return wordsList

    def __get_files_list(self, path: str,  endswith: str) -> list:
        """Finds files with .polarity extension in the desired path"""

        list_of_files = [path + f for f
                         in listdir(path)
                         if isfile(join(path, f)) and
                         f.endswith(endswith)]
        return list_of_files

    def __calculate_lines(self) -> int:
        # Get the list of all files in folder
        f = open("data/all_tweets.txt", "w")
        self.filesList = self.__get_files_list(
            self.__dataset_path, ".polarity")
        for file in self.filesList:
            count = 0
            file_read = open(file, 'r')
            try:
                lines = file_read.readlines()
                for line in lines:
                    count += 1
                    f.writelines(PrepareData.clean_string(line))
                    f.writelines("\n")
            except Exception as e:
                print(e)

            if "data/agreed.polarity" == file:
                agr_lines = count
            else:
                dis_lines = count
            file_read.close()
        f.close()

        agr_and_disagree = [agr_lines, dis_lines]
        with open("data/length.pckl", 'wb') as f:
            pickle.dump(agr_and_disagree, f)

        self.__overall_line_number = agr_lines + dis_lines
        return agr_lines, dis_lines

    def __check_idx_matrix_occurance(self) -> None:
        """Checks if any idx matrix exists"""

        rnn = RNNModel()
        rnn.set_agr_lines, rnn.set_dis_lines = self.__calculate_lines()
        rnn.set_overall_lines = self.__overall_line_number

        idsMatrix = self.__get_files_list(self.__dataset_path, "idsMatrix.npy")
        if len(idsMatrix) >= 1:
            ans = input(
                "Found 'idsMatrix'. Would you like to recreate it? (y/n) ")
            if ans in ["y", "", "Yes", "Y"]:
                self.__create_idx()
            else:
                print("Continue...")
        else:
            print("Haven't found the idx matrix models.")
            self.__create_idx()
        rnn.create_and_train_model()

    def __create_idx(self) -> None:
        """Function of idx creation"""
        wordsList = self.__get_words_list()
        ids = np.zeros((self.__overall_line_number + 1, self.__maxSeqLength),
                       dtype='int32')
        with open("data/all_tweets.txt", "r") as f:
            lines = f.readlines()
        for line_mumber, line in enumerate(lines):
            if line_mumber % 1000 == 0:
                print(
                    f"Reading line number: \
                    {line_mumber}/{self.__overall_line_number}")

            cleaned_line = self.clean_string(line)
            splitted_line = cleaned_line.split()
            for w_num, word in enumerate(splitted_line):
                try:
                    get_word_index = wordsList.index(word)
                    ids[line_mumber][w_num] = get_word_index
                except ValueError:
                    ids[line_mumber][w_num] = 399999
                if w_num >= self.__maxSeqLength - 1:
                    break
        np.save('data/idsMatrix', ids)
        print("Saved ids matrix to the 'model/idsMatrix';")


class RNNModel():
    """Class of TF models creation"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Avoid the tf warnings

    def __init__(self):
        self.__batchSize = config.batchSize
        self.__lstmUnits = config.lstmUnits
        self.__numClasses = config.numClasses
        self.__numDimensions = config.numDimensions
        self.__maxSeqLength = config.maxSeqLength
        self.__wordVectors = np.load('data/wordVectors.npy')
        self.__agr_lines = int
        self.__dis_lines = int
        self.__overall_lines = int
        self.learning_rate = config.learning_rate

    @property
    def get_agr_lines(self):
        return self.__agr_lines

    @get_agr_lines.setter
    def set_agr_lines(self, value):
        self.__agr_lines = value

    @property
    def get_dis_lines(self):
        return self.__dis_lines

    @get_dis_lines.setter
    def set_dis_lines(self, value):
        self.__dis_lines = value

    @property
    def get_overall_lines(self):
        return self.__overall_lines

    @get_overall_lines.setter
    def set_overall_lines(self, value):
        self.__overall_lines = value

    def __get_train_batch(self) -> list:
        """Returning training batch function"""
        labels = []
        arr = np.zeros([self.__batchSize, self.__maxSeqLength])
        for i in range(self.__batchSize):
            if i % 2 == 0:
                num = randint(1, int(self.__agr_lines * 0.9))
                labels.append([1, 0])  # Agreed
            else:
                from_line = int(self.__agr_lines +
                                (self.__dis_lines * 0.1)) + 1
                to_line = int(self.__overall_lines)
                num = randint(from_line, to_line)
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num]
        return arr, labels

    def __get_test_batch(self) -> list:
        """Returning training batch function"""
        labels = []
        with open('data/length.pckl', 'rb') as f:
            agr_lines, dis_lines = pickle.load(f)

        arr = np.zeros([self.__batchSize, self.__maxSeqLength])

        agr_from_line = int(agr_lines * 0.9) + 1
        agr_to_line = agr_lines
        dis_from_line = agr_lines + 1
        dis_to_line = int(agr_lines + (dis_lines * 0.1)) + 1

        for i in range(self.__batchSize):
            if i % 2 == 0:
                num = randint(agr_from_line, agr_to_line)
                labels.append([1, 0])  # Agreed
            else:
                num = randint(dis_from_line, dis_to_line)
                labels.append([0, 1])  # Disagreed
            arr[i] = self.ids[num]
        return arr, labels

    def create_and_train_model(self) -> None:
        """Creates the TF model"""
        self.ids = np.load('data/idsMatrix.npy')
        print("Creating training model...")
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        labels = tf.placeholder(tf.float32,
                                [self.__batchSize, self.__numClasses])
        tf.add_to_collection("labels", labels)

        input_data = tf.placeholder(tf.int32,
                                    [self.__batchSize, self.__maxSeqLength])
        # We are saving to the collections, in order to resore it later
        tf.add_to_collection("input_data", input_data)

        data = tf.Variable(tf.zeros([self.__batchSize,
                                     self.__maxSeqLength,
                                     self.__numDimensions]), dtype=tf.float32)

        data = tf.nn.embedding_lookup(self.__wordVectors, input_data)
        cells = []
        for _ in range(config.cells):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.__lstmUnits)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                cell=lstm_cell,
                output_keep_prob=0.75
            )
            cells.append(lstm_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        initial_state = cell.zero_state(self.__batchSize, tf.float32)
        value, final_state = tf.nn.dynamic_rnn(cell, data,
                                               initial_state=initial_state,
                                               dtype=tf.float32)

        weight = tf.Variable(tf.truncated_normal(
            [self.__lstmUnits,
             self.__numClasses])
        )
        bias = tf.Variable(tf.constant(0.1, shape=[self.__numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        # Adding prediction to histogram
        tf.summary.histogram('predictions', prediction)

        # Here we are doing the same
        tf.add_to_collection("prediction", prediction)
        tf.add_to_collection("max_seq_length", self.__maxSeqLength)
        tf.add_to_collection("batch_size", self.__batchSize)

        correct_pred = tf.equal(tf.argmax(prediction, 1),
                                tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.add_to_collection("accuracy", accuracy)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=prediction, labels=labels)
        )
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)
        tf.add_to_collection("optimizer", optimizer)
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.histogram("Out", value[:, -1])
        merged = tf.summary.merge_all()

        # ------ Below is training process ---------
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "models/" + str(folder_name) + "/"
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        with open(f"{log_dir}configs.txt", 'w') as f:
            f.write("Number of dimensions: {}\n".format(config.numDimensions))
            f.write("Sequence length: {}\n".format(config.maxSeqLength))
            f.write("Batch sizes: {}\n".format(config.batchSize))
            f.write("LSTM units: {}\n".format(config.lstmUnits))
            f.write("Number of classes: {}\n".format(config.numClasses))
            f.write("Cells: {}\n".format(config.cells))
            f.write("Training steps: {}\n".format(config.training_steps))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(config.training_steps+1):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.__get_train_batch()
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
                    self.__batchSize, tf.float32))
                nextBatch, nextBatchLabels = self.__get_test_batch()
                feed = {input_data: nextBatch,
                        labels: nextBatchLabels,
                        initial_state: val_state}
                summary, batch_acc, val_state = sess.run(
                    [merged, accuracy, final_state], feed_dict=feed)
                val_acc.append(batch_acc)
                avg_acc = np.mean(val_acc)
                print("\nVal acc: {:.3f}\n".format(avg_acc))
            # Save the network every 10,000 training iterations
            # if (i % 1000 == 0 and i != 0):
            #     save_path = saver.save(sess,
            #                            "models/pretrained_lstm.ckpt",
            #                            global_step=i)
            #     print(f"Saved to {save_path}")
        save_path = f"{log_dir}pretrained_lstm.ckpt"
        saver.save(sess, save_path, global_step=config.training_steps)
        print(f"Model saved to: {save_path}")
        writer.close()
        sess.close()

    def test_model(self, dir_) -> None:
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
                nextBatch, nextBatchLabels = self.__get_test_batch()
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
