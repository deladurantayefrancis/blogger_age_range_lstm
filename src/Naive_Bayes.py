import numpy as np
import sys

from utils import *


class NaiveBayes:

    def __init__(self, classes, priors):
        """
        Instantiates a Naive Bayes model which uses frequency analysis within the data to calculate the conditional
        probabilities
        :param classes: List of all possible classes that are represented by the model
        :param priors: Prior knowledge of the probability of each possible class within the model
        """

        # By definition, we always have as many priors as classes
        assert len(priors) == len(classes), "len(priors) != len(classes)"

        self.m, self.classes = len(classes), classes
        self.priors = priors

        self.all_words = None

        self.base_vocab = None  # to avoid preprocessing training data multiple times

        self.probs_by_class = []

    def get_vocabulary(self, train_data, threshold):
        """
        From a list of preprocessed comments, builds a dictionnary of all of the words that can appear within a comment.
        :param train_data: List of preprocessed comments (that is, comments with no stopwords, url or
        punctuation and that have been stemmed) from which to extract the vocabulary
        :param threshold: Threshold indicating if a word is to be considered part of the vocabulary. A word that apears
        a number of times inferior to the threshold is not included in the dictionary
        :return: Returns a set of all unique words found in the comments which appear a number of times equal or
        superior to the threshold
        """

        # We only extract the base vocabulary if we haven't done so already.
        if self.base_vocab is None:
            logger("extracting base vocabulary...")

            # retrieve words from preprocessed training data
            self.all_words = ' '.join(train_data).split()

            # compute base vocabulary word frequencies
            self.base_vocab = dict.fromkeys(self.all_words, 0)
            for word in self.all_words:
                self.base_vocab[word] += 1

        logger("getting vocabulary...")

        # replace infrequent words (< threshold) with an UNK token
        if threshold > 1:
            vocabulary = dict({word: count for word, count in self.base_vocab.items() if count >= threshold})
            vocabulary['UNK'] = len(self.all_words) - np.sum(list(vocabulary.values()))
        else:
            # If there's no threshold, we keep the vocabulary as is
            vocabulary = self.base_vocab.copy()
            vocabulary['UNK'] = 0

        # return vocabulary
        return set(vocabulary)

    def train(self, train_data, train_labels, threshold, smoothing):
        """
        Function that trains a Naive Bayes model by learning all of the word frequencies for each class.
        :param train_data: Training set of preprocessed comments (that is, comments with no stopwords,
        url or punctuation and that have been stemmed)
        :param train_labels: Labels associated with each comment
        :param threshold: Threshold under which a word is not considered part of the dictionary
        :param smoothing: Smoothing constant used for Laplace smoothing on the data
        """
        # get vocabulary for current threshold
        vocabulary = self.get_vocabulary(train_data, threshold)

        logger("training...")

        # will hold per class vocabulary word probabilities
        self.probs_by_class = []

        # For each class we build a dictionary of word frequencies
        for c in self.classes:
            # initialize word frequencies to zero
            word_freqs = dict.fromkeys(vocabulary, 0)
            word_freqs['UNK'] = 0
            
            for data in minibatch(train_data[train_labels == c], size=batch_size):
                # extract and preprocess class data. See Data_Processing.py
                class_data = ' '.join(data)
                class_words = class_data.split()

                # compute word frequencies
                words, freqs = np.unique(class_words, return_counts=True)
                for word, freq in zip(words, freqs):
                    if word in vocabulary:
                        word_freqs[word] += freq
                    else:
                        word_freqs['UNK'] += freq
                
            # compute word probabilities with Laplace smoothing
            smoothing_total = np.sum(list(word_freqs.values())) + (smoothing * len(vocabulary))
            word_probs = {word: (freq + smoothing) / smoothing_total for word, freq in word_freqs.items()}

            # save computed word probabilities for the class
            self.probs_by_class.append(word_probs)

    def compute_logits(self, test_data):
        """
        Given a set of preprocessed comments and no labels, proceeds to calculate the log probability that each comment
        will be in a given class
        :param test_data: Testing set of preprocessed comments (that is, comments with no stopwords,
        url or punctuation and that have been stemmed)
        :return: Matrix of log probabilities M where M(i,c) is the log probability that comment i is in class c
        """
        logger("predicting...")

        # will hold class log-probabilities (logits) of the posts
        logits = np.zeros((len(test_data), self.m))

        # For every post in the test set
        for i, post in enumerate(test_data):

            # preprocess current post
            post_words = str(post).split()

            # compute log-probabilities
            for c in range(self.m):
                logits[i, c] = np.sum([np.log(self.probs_by_class[c][w])
                    if w in self.probs_by_class[c] else np.log(self.probs_by_class[c]['UNK'])
                        for w in post_words])
        
        # add log of class priors
        logits += np.log(self.priors)

        return logits


def model_search(train_data, train_labels, valid_data, valid_labels,
                 classes, priors, one_vs_all):
    """
    Function that finds the optimal hyperparameters (smoothing and threshold) using a basic gridsearch algorithm over a
    subspace of the hyperparametric space
    :param train_data: List of preprocessed comments used for training
    :param train_labels: List of labels associated with each comment on the training set
    :param valid_data: List of preprocessed comments used for validation ahd hyperparameter search
    :param valid_labels: List of labels associated with each comment on the validation set
    :param classes: List of all classes represented in the model
    :param one_vs_all: Whether we are training a single naive bayes model for all classes or one model for a one
    vs all method
    :return: Bayes model trained on the training set with the optimal hyperparameters defined on the validation set
    """

    # class count
    m = len(classes)

    # model instantiation
    model = NaiveBayes(classes, priors)

    # hyperparameter grid search
    thresholds = np.arange(2) + 1
    smoothings = np.concatenate(([.001, .005, .01, .05], (np.arange(10) + 1) / 10))  # Laplace smoothing
    
    ova_accuracy = None

    best_accuracy = 0
    best_ova_accuracy = 0
    best_threshold = None
    best_smoothing = None

    for i, threshold in enumerate(thresholds):
        for j, smoothing in enumerate(smoothings):
            # current hyperparameter combination
            logger(f'\n--- threshold={threshold}, smoothing={smoothing} ---')

            # train, predict and compute accuracy
            model.train(train_data, train_labels, threshold, smoothing)
            valid_logits = model.compute_logits(valid_data)
            valid_preds = get_pred_classes(classes, valid_logits)
            accuracy = compute_accuracy(valid_preds, valid_labels)
            logger(f'Validation accuracy: {accuracy}')

            # If we're in a OVA model, we compute the accuracy of the model one the class it was trained to predict
            if one_vs_all:
                ova_accuracy = compute_accuracy(
                    valid_preds[valid_labels == classes[0]],
                    valid_labels[valid_labels == classes[0]])
                logger(f'"{classes[0]}" accuracy: {ova_accuracy}')

            # if best accuracy until now, save as best with hyperparameters
            if one_vs_all and accuracy+ova_accuracy > best_accuracy+best_ova_accuracy:
                best_ova_accuracy = ova_accuracy
                best_accuracy = accuracy
                best_threshold = threshold
                best_smoothing = smoothing
            elif not one_vs_all and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_smoothing = smoothing
                

    logger(f'\nBest validation accuracy: {best_accuracy}')
    if one_vs_all:
        logger(f'Best "{classes[0]}" accuracy: {best_ova_accuracy}')
    logger(f'Best hyperparameters: threshold={best_threshold}, smoothing={best_smoothing}\n')

    # retrain the model on train and valid set using best hyperparameters
    model.train(
        np.concatenate((train_data, valid_data)),
        np.concatenate((train_labels, valid_labels)),
        best_threshold, 
        best_smoothing
    )

    return model


if __name__ == "__main__":

    # number of inputs to train on
    num_inputs = int(sys.argv[1])

    # comment about the current run
    run_description = sys.argv[2] if len(sys.argv) > 2 else None

    # batch size
    batch_size = 1024

    # training mode
    one_vs_all = True

    # proportion of the full dataset to be used for validation
    valid_prop = .1

    # proportion of the full dataset to be used for accuracy testing
    test_prop = .05

    # Checking if the required folders exist. See utils.py
    fs_check()

    # load and split datasets
    logger('loading datasets...')
    train_valid, submission_data = load_data(
        num_inputs,
        train_dataset='data_train_preprocessed.csv',
        submission_dataset='data_test_preprocessed.csv'
    )

    if num_inputs == -1:
        num_inputs = len(train_valid)
    
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = \
        split_data(train_valid, valid_prop, test_prop)
    logger('loading complete!')

    """
    train_data = train_data[:10000]
    train_labels = train_labels[:10000]
    """

    # Get the list of classes
    classes = list(set(train_labels))
    print(classes)
    print(len(classes))
    classes.sort()


    labels, counts = np.unique(train_valid[classe], return_counts=True)
    priors = counts / np.sum(counts)
    print('priors:', priors * 100)
    assert np.abs(np.sum(priors) - 1) < .001, "priors should be a probability distribution"


    logger(f'Description: {run_description}')
    logger(f'one_vs_all: {one_vs_all}')
    logger(f'num_inputs: {num_inputs}')
    logger(f'valid_prop: {100*valid_prop}%')
    logger(f'test_prop: {100*test_prop}%')


    # training the models in one vs all method
    if one_vs_all:

        # Matrix containing the probabilities of each entries being in each class for the test set and the
        # submission set. The submission set is the set of inputs on which the submission predictions are made.
        test_probs_vs_all = np.zeros((len(test_data), len(classes)))
        submission_probs_vs_all = np.zeros((len(submission_data), len(classes)))

        for i, c in enumerate(classes):
            logger(f'\n==> "{c}" versus all! <==')

            # make one vs all labels for training and validation sets
            t_labels = train_labels.copy()
            t_labels[t_labels != c] = 4
            v_labels = valid_labels.copy()
            v_labels[v_labels != c] = 4

            # search for best 'class vs all' model on validation set
            optimal_model = model_search(
                train_data, t_labels,
                valid_data, v_labels,
                [c, 4], [priors[i], 1 - np.sum(priors[i])],
                one_vs_all
            )

            # get 'per class' logits and 'class vs all' probs on test set 
            test_logits = optimal_model.compute_logits(test_data)
            test_probs_vs_all[:, i] = softmax(test_logits, axis=1)[:, 0]

            # get 'per class' logits and 'class vs all' probs on submission set
            submission_logits = optimal_model.compute_logits(submission_data)
            submission_probs_vs_all[:, i] = softmax(submission_logits, axis=1)[:, 0]

        # take class with highest probability vs all others as the prediction and compute accuracy
        test_preds = get_pred_classes(classes, test_probs_vs_all)
        accuracy = compute_accuracy(test_preds, test_labels)
        logger(f'\nTest accuracy: {accuracy}')

        # take class with highest probability vs all others as the prediction save to file
        submission_preds = get_pred_classes(classes, submission_probs_vs_all)
        logger("Writing submission predictions to file...")
        generate_submission(submission_preds, f'results/{one_vs_all}_{accuracy}_{num_inputs}.csv')

    # Training a single statistical model for all classes
    else:
        # search for model with best hyperparameters on validation set
        optimal_model = model_search(
            train_data, train_labels,
            valid_data, valid_labels,
            classes, priors,
            one_vs_all
        )

        # make predictions and compute accuracy on test set
        test_logits = optimal_model.compute_logits(test_data)
        test_preds = get_pred_classes(classes, test_logits)
        accuracy = compute_accuracy(test_preds, test_labels)
        logger(f'Test accuracy: {accuracy}')

        # make predictions and save to file on submission set
        submission_logits = optimal_model.compute_logits(submission_data)
        submission_preds = get_pred_classes(classes, submission_logits)
        logger("Writing submission predictions to file...")
        generate_submission(submission_preds, f'results/{one_vs_all}_{accuracy}_{num_inputs}.csv')

    # compute confusion matrix
    conf_matrix = confusion_matrix(classes, test_preds, test_labels, pourcents=False)
    logger(f'\nClasses:\n{classes}')
    logger(f'\nConfusion matrix on test set:\n{conf_matrix}')
    
    binary_classifiers = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            if conf_matrix[i][j] >= 10:
                binary_classifiers.append((i, j))
    