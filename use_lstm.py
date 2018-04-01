import os
from string import punctuation

import numpy as np
import tensorflow as tf

import config


class TryLstm():
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.maxSeqLength = config.maxSeqLength
        self.batchSize = config.batchSize
        self.load_gloves()
        self.restore_models()

    def load_gloves(self):
        print("Loading gloves model...")
        self.wordsList = np.load('data/wordsList.npy').tolist()
        self.wordsList = [word for word in self.wordsList]

    def restore_models(self):
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Restoring the meta and latest model
        path = ".".join([tf.train.latest_checkpoint("models/"), "meta"])
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.sess, tf.train.latest_checkpoint('models/'))

        # Restoring the tf variables
        self.input_data = tf.get_collection("input_data")[0]
        self.prediction = tf.get_collection("prediction")[0]

    def predict(self, inputText):
        inputMatrix = self.getSentenceMatrix(inputText)
        predictedSentiment = self.sess.run(self.prediction,
                                           {self.input_data: inputMatrix}
                                           )[0]
        if (predictedSentiment[0] > predictedSentiment[1]):
            print("|----------------------------------------------------|")
            print("|---The comment message has agreement sentiment------|")
            print("|----------------------------------------------------|")
        else:
            print("|----------------------------------------------------|")
            print("|---The comment message has disagreement sentiment---|")
            print("|----------------------------------------------------|")
        print(f"Agreement coefficient:",
              "{0:.2f}".format(predictedSentiment[0]))
        print(f"Disagreement coefficient:",
              "{0:.2f}".format(predictedSentiment[1]))

    def clean_sentence(self, string):
        string = string.lower()
        cleaned_string = ''
        for num, char in enumerate(string):
            # Ignoring the "< - >"" statement
            if char == "<":
                if string[num + 2] == "-" and string[num + 4] == ">":
                    cleaned_string += char
            elif char == "-":
                if string[num - 2] == "<" and string[num + 2] == ">":
                    cleaned_string += char
            elif char == ">":
                if string[num - 4] == "<" and string[num - 2] == "-":
                    cleaned_string += char
            # Deleting the punctuation marks
            elif char not in punctuation:
                cleaned_string += char
        return cleaned_string

    def getSentenceMatrix(self, sentence):
        sentenceMatrix = np.zeros([self.batchSize, self.maxSeqLength],
                                  dtype='int32')
        cleanedSentence = self.clean_sentence(sentence)
        split = cleanedSentence.split()
        for indexCounter, word in enumerate(split):
            try:
                sentenceMatrix[0, indexCounter] = self.wordsList.index(word)
            except ValueError:
                # Vector for unknown words
                sentenceMatrix[0, indexCounter] = 000
        return sentenceMatrix


if __name__ == '__main__':
    var = TryLstm()
    try:
        while True:
            original_tweet = input("\n---Enter origin message: ")
            if original_tweet == "exit":
                var.sess.close()
                exit(1)
            comment = input("---Enter comment message: ")
            inputText = original_tweet + " < - > " + comment
            var.predict(inputText)
    except Exception as e:
        print("Exception occurred: ", e)
    finally:
        # Close the session and exit the code in the end
        var.sess.close()
        exit(1)
