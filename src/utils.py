import numpy as np
from datetime import datetime
import os
import pandas as pd

log_file = None

blog = 'blog'
classe = 'classe'


# Batcher function
def minibatch(iterable1, iterable2=None, size=1):
    l = len(iterable1)
    n = size
    for ndx in range(0, l, n):
        index2 = min(ndx + n, l)
        if iterable2 is None:
            yield iterable1[ndx: index2]
        else:
            yield iterable1[ndx: index2], iterable2[ndx: index2]


def confusion_matrix(classes, preds, labels, pourcents=False):
    """
    Generates a confusion matrix indicated which class is often mixed up with which other class
    :param classes: List of all possible classes in the model
    :param preds: Classification predictions given by a model
    :param labels: True labels for the given predictions
    :param pourcents: Whether or not to express the values in the confusion matrix in percentages
    :return: A m X m confusion matrix M where M(i,j) is how many times, on average, a point belonging to class i is
    given class j
    """
    m = len(classes)
    conf_matrix = np.zeros((m, m))
    for i, ground_truth in enumerate(classes):
        class_idx = (labels == ground_truth)
        for j, predicted in enumerate(classes):
            if pourcents:
                conf_matrix[i, j] = np.round(100 * np.mean(preds[class_idx] == predicted), 2)
            else:
                conf_matrix[i, j] = np.mean(preds[class_idx] == predicted)

    return conf_matrix


def softmax(logits, axis=None):
    """
    Softmax function used to rescale the logits and prevent numerical overflow
    :param logits: Array of logits to rescale.
    :param axis: Axis on which to calculate the probabilities
    :return:
    """
    rescaled = logits - np.max(logits, axis=axis, keepdims=True)
    e_logits = np.e ** rescaled
    probs = e_logits / np.sum(e_logits, axis=axis, keepdims=True)

    return probs


def compute_accuracy(preds, labels):
    """
    Compute the accuracy of a model based on a list of predictions from that model and a list of ground-truths for
    those predictions
    :param preds: Predictions given by a statistical learning model
    :param labels: Ground truths for each prediction of the model
    :return: Percentage (expressed from o to 1) of accuracy of the model
    """
    return np.mean(preds == labels)


def logger(str_to_log):
    """
    Logging function used to write to file the accuracy of the trained models during training and testing
    :param str_to_log: String that will be written to the log file
    """
    global log_file
    if log_file is None:
        log_file = 'logs/' + datetime.now().strftime('%Y-%m-%d__%H-%M-%S') + '.txt'

    with open(log_file, 'a') as log:
        log.write(str_to_log + '\n')
        print(str_to_log)


def split_data(dataset, valid_prop, test_prop):
    """
    Function that takes a dataset and splits it into three subsets: a training set, a validation set, and a test set
    :param dataset: Complete dataset to split into training validation and test sets
    :param valid_prop: What proportion (in percentage; expressed as a value from 0 to 1) of the full dataset should be
    used for the validation set
    :param test_prop: What proportion (in percentage; expressed as a value from 0 to 1) of the full dataset should be
    used for the test set
    :return: A tuple containing, in that order:
        * The datapoints of the training set
        * The label of each datapoint in the training set
        * The datapoints of the validation set
        * The label of each datapoint in the validation set
        * The datapoints of the test set
        * The label of each datapoint in the test set
    """

    assert valid_prop > 1 and test_prop > 1 or valid_prop < 1 and test_prop < 1, \
        "valid_prop / test_prop error"


    if  valid_prop > 1:
        valid_split_idx = valid_prop
        test_split_idx = valid_prop + test_prop
    else:
        n = len(dataset[blog])
        valid_split_idx = int(valid_prop * n)
        test_split_idx = int((valid_prop + test_prop) * n)

    #train_data, train_labels = np.asarray(dataset[0][test_split_idx:]), np.asarray(dataset[1][test_split_idx:])
    #valid_data, valid_labels = np.asarray(dataset[0][:valid_split_idx]), np.asarray(dataset[1][:valid_split_idx])
    #test_data, test_labels = np.asarray(dataset[0][valid_split_idx:test_split_idx]), \
    #    np.asarray(dataset[1][valid_split_idx: test_split_idx])

    train_data, train_labels = np.asarray(dataset[blog][test_split_idx:]), np.asarray(dataset[classe][test_split_idx:])
    valid_data, valid_labels = np.asarray(dataset[blog][:valid_split_idx]), np.asarray(dataset[classe][:valid_split_idx])
    test_data, test_labels = np.asarray(dataset[blog][valid_split_idx:test_split_idx]), \
        np.asarray(dataset[classe][valid_split_idx: test_split_idx])

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


def generate_submission(predictions, filename='test_predictions.csv'):
    """
    Function that generates the output predictions file used in the kaggle competition.
    :param predictions: List of label predictions given by a model of all the comments found in data_test.pkl
    :param filename: Name of the file in which to write the predictions
    """
    with open(filename, mode='w+') as f:
        f.write("Id,Category")
        for i, pred in enumerate(predictions):
            f.write(f'\n{i},{pred}')


def get_pred_classes(classes, probs_or_logits):
    """
    From a list of probabilities or logits for each input comment, return the predicted label for that comment
    :param classes: List of all possible class labels
    :param probs_or_logits: Array of probabilities or logits for each class and each comment.
    :return: Returns a list of labels corresponging to the most probable class label for every given comment.
    """
    preds_idx = np.argmax(probs_or_logits, axis=1)
    preds = list(map(lambda idx: classes[idx], preds_idx))
    return np.asarray(preds)


def load_data(num_inputs, train_dataset, submission_dataset, folder='./out/'):
    """
    Unpacks content from both data_train.pkl and data_test.pkl in numpy arrays and returns those arrays.
    :param folder: Folder in which data_train.pkl and data_test.pkl can be found
    :return: A tuple containing the training data and testing data recovered from the files
    """

    train_set = pd.read_csv(folder + train_dataset, names=[blog,classe])
    submission_set = pd.read_csv(folder + submission_dataset, names=[blog,classe])
    #train_set = #np.load(folder + filename)
    # Added x to name to prevent shadow from outer scope.
    #test_data_x = np.load(folder + filename)
    if num_inputs != -1:
        train_set = train_set.sample(num_inputs, random_state=1234)
    else:
        train_set = train_set.sample(frac=1, random_state=1234)

    return train_set, submission_set[blog]


def fs_check():
    """
    Checks is the folders necessary for logging exist and, if not, creates them.
    """
    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isdir('logs'):
        os.mkdir('logs')

    if not os.path.isdir('out'):
        os.mkdir('out')

    if not os.path.isdir('results'):
        os.mkdir('results')
