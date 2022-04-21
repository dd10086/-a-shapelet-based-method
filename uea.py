import os
import json
import math

import numpy as np
import torch
import numpy
import argparse

import timeit
import wrappers
from sklearn.model_selection import train_test_split
import sktime
from sktime.utils.data_io import load_from_tsfile_to_dataframe, load_from_arff_to_dataframe
from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
    from_multi_index_to_3d_numpy,
    from_nested_to_3d_numpy,
)
from sktime.datatypes._panel._check import is_nested_dataframe


def convert_numpy(data):
    data_list = []
    for i in range(data.shape[0]):
        tmp = []
        for j in range(data.shape[1]):
            tmp.append(data[i, j].values)
        data_list.append(numpy.array(tmp))
    return numpy.array(data_list)


def load_UEA_dataset(path, dataset):
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.ts")
    test_file = os.path.join(path, dataset, dataset + "_TEST.ts")
    train, train_labels = load_from_tsfile_to_dataframe(train_file)
    test, test_labels = load_from_tsfile_to_dataframe(test_file)
    train = from_nested_to_3d_numpy(train)
    test = from_nested_to_3d_numpy(test)
    nb_dims = train.shape[1]
    # Normalizing dimensions independently
    for j in range(nb_dims):
        mean = numpy.mean(numpy.concatenate((train[:, j], test[:, j]), axis=0))
        var = numpy.var(numpy.concatenate((train[:, j], test[:, j]), axis=0))
        train[:, j] = (train[:, j] - mean) / math.sqrt(var)
        test[:, j] = (test[:, j] - mean) / math.sqrt(var)

    labels = numpy.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i
    train_labels = numpy.vectorize(transform.get)(train_labels)
    test_labels = numpy.vectorize(transform.get)(test_labels)
    tmp = np.concatenate((train, test))
    tmp_label = np.concatenate((train_labels, test_labels))
    train, test, train_labels, test_labels = train_test_split(tmp, tmp_label, test_size=0.2)

    print('dataset load succeed !!!')
    return train, train_labels, test, test_labels


def fit_parameters(file, train, train_labels, test, test_labels, cuda, gpu, save_path, cluster_num, node_number,
                   save_memory=False):
    """
    Creates a classifier from the given set of parameters in the input
    file, fits it and return it.
    """
    classifier = wrappers.CausalCNNEncoderClassifier()
    # classifier.
    # Loads a given set of parameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['in_channels'] = 1
    params['cuda'] = cuda
    params['gpu'] = gpu
    params['Adj'] = torch.zeros(node_number, node_number, dtype=torch.float64)
    classifier.set_params(**params)
    return classifier.fit(
        train, train_labels, test, test_labels, save_path, cluster_num, node_number, save_memory=save_memory,
        verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', default='BasicMotions', type=str, metavar='D', required=False,
                        help='dataset name')
    parser.add_argument('--path', default='./datasets', type=str, metavar='PATH', required=False,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', default='./estimator results', type=str, metavar='PATH', required=False,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', default=0, action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', default='default_parameters.json', type=str, metavar='FILE', required=False,
                        help='path of the file of parameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the model and retrain the classifier')
    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False
    train, train_labels, test, test_labels = load_UEA_dataset(
        args.path, args.dataset
    )
    node_number = 105
    cluster_num = 200
    if not args.load and not args.fit_classifier:
        print('start new network training')
        classifier = fit_parameters(
            args.hyper, train, train_labels, test, test_labels, args.cuda, args.gpu, args.save_path, cluster_num,
            node_number,
            save_memory=False
        )
    else:
        classifier = wrappers.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        classifier.set_params(**hp_dict)
        classifier.load(os.path.join(args.save_path, args.dataset))

    if not args.load:
        if args.fit_classifier:
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(
            os.path.join(args.save_path, args.dataset)
        )
        with open(
                os.path.join(
                    args.save_path, args.dataset + '_parameters.json'
                ), 'w'
        ) as fp:
            json.dump(classifier.get_params(), fp)
    end = timeit.default_timer()
    print("All time: ", (end - start) / 60)
    print('finished')
