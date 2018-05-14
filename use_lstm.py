import os
import argparse

import numpy as np
import tensorflow as tf

from train_and_test import PrepareData


class TryLstm():
    def __init__(self, path):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.__path = path
        self.__load_gloves()
        self.__restore_models()

    def __load_gloves(self):
        print("Loading gloves model...")
        self.wordsList = np.load('data/wordsList.npy').tolist()
        self.wordsList = [word for word in self.wordsList]

    def __restore_models(self):
        tf.reset_default_graph()
        self.sess = tf.Session()

        # Restoring the meta and latest model
        path = ".".join([tf.train.latest_checkpoint(self.__path), "meta"])
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.__path))

        # Restoring the tf variables
        self.input_data = tf.get_collection("input_data")[0]
        self.prediction = tf.get_collection("prediction")[0]
        self.__maxSeqLength = tf.get_collection("max_seq_length")[0]
        self.__batchSize = tf.get_collection("batch_size")[0]

    def predict(self, inputText):
        inputMatrix = self.__getSentenceMatrix(inputText)
        predictedSentiment = self.sess.run(self.prediction,
                                           {self.input_data: inputMatrix}
                                           )[0]
        if predictedSentiment[0] > predictedSentiment[1]:
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

    def __getSentenceMatrix(self, sentence):
        sentenceMatrix = np.zeros([self.__batchSize, self.__maxSeqLength],
                                  dtype='int32')
        cleanedSentence = PrepareData.clean_string(sentence)
        split = cleanedSentence.split()
        for indexCounter, word in enumerate(split):
            if indexCounter >= self.__maxSeqLength:
                break
            try:
                sentenceMatrix[0, indexCounter] = self.wordsList.index(word)
            except ValueError:
                # Vector for unknown words
                sentenceMatrix[0, indexCounter] = 399999
        return sentenceMatrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Use the model")
    args = parser.parse_args()
    var = TryLstm(args.model) if args.model else None
    if var is not None:
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
    else:
        raise FileNotFoundError("The model path hasn't been provided")
