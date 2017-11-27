"""
Base code: https://github.com/uglyboxer/affdigits/blob/master/affnist_read.py
"""
import os
import glob
import re
import numpy as np
import scipy.io as spio


def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


class AffNistBatchLoader(object):
    def __init__(self, training_dir, n_sets=5, one_hot=True):
        if n_sets <= 0 or n_sets > 32:
            raise ValueError('There are 1 ~ 32 sets, each contains 50,000 examples.')

        # set affNIST data attribute infomations
        im_size = 40
        single_train_size = 50000

        # get matlab *.mat files
        training_batches = glob.glob(os.path.join(training_dir, '*.mat'))
        if len(training_batches) == 0:
            raise ValueError('Cannot find any *.mat files')

        # sort for convinience
        training_batches.sort(key=natural_keys)

        # set class attributes
        self.n_sets = n_sets
        self.one_hot = one_hot
        self.num_examples = self.n_sets * single_train_size

        # merge training examples
        self.images = np.empty((self.num_examples, im_size * im_size), dtype=np.float32)
        self.labels = np.empty((self.num_examples, 1), dtype=np.float32)
        if self.one_hot:
            self.labels = np.empty((self.num_examples, 10), dtype=np.float32)

        # iterate all examples and merge
        for ii in range(self.n_sets):
            train_mat_fn = training_batches[ii]
            train_dataset = loadmat(train_mat_fn)

            # load image and label
            x = train_dataset['affNISTdata']['image'].transpose().astype(np.float32)
            y = train_dataset['affNISTdata']['label_int'].transpose().astype(np.float32)
            if self.one_hot:
                y = train_dataset['affNISTdata']['label_one_of_n'].transpose()

            # normalize image to 0 ~ 1
            x = x / 255.0

            # merge
            start = ii * single_train_size
            end = (ii + 1) * single_train_size
            self.images[start:end] = x
            self.labels[start:end] = y

        # reset batch index
        self.batch_index = 0

        return

    def next_batch(self, batch_size):
        if self.batch_index + batch_size > self.num_examples:
            self.batch_index = 0

        x = self.images[self.batch_index:self.batch_index + batch_size]
        y = self.labels[self.batch_index:self.batch_index + batch_size]

        self.batch_index += batch_size
        return x, y


class AffNistLoader(object):
    def __init__(self, mat_file, one_hot=True):
        # set class attributes
        self.one_hot = one_hot

        # load mat file
        dataset = loadmat(mat_file)

        # load image and label
        self.images = dataset['affNISTdata']['image'].transpose().astype(np.float32)
        self.labels = dataset['affNISTdata']['label_int'].transpose().astype(np.float32)
        if self.one_hot:
            self.labels = dataset['affNISTdata']['label_one_of_n'].transpose()

        # normalize image to 0 ~ 1
        self.images = self.images / 255.0

        # find number of samples
        self.num_examples = self.labels.shape[0]

        # reset batch index
        self.batch_index = 0

        return

    def next_batch(self, batch_size):
        if self.batch_index + batch_size > self.num_examples:
            self.batch_index = 0

        x = self.images[self.batch_index:self.batch_index + batch_size]
        y = self.labels[self.batch_index:self.batch_index + batch_size]

        self.batch_index += batch_size
        return x, y


if __name__ == '__main__':
    from scipy.misc import toimage
    from utils import form_image

    n_test_case_block = 5
    n_test_case = n_test_case_block * n_test_case_block

    # check train loader
    trainig_dir = './affNIST/training_batches'
    aff_mnist_train_loader = AffNistBatchLoader(trainig_dir, n_sets=3, one_hot=True)

    # get training data
    train_batch_x, train_batch_y = aff_mnist_train_loader.next_batch(n_test_case)

    # reshape input
    train_batch_x = np.reshape(train_batch_x, (-1, 40, 40, 1))

    train_image = form_image(train_batch_x, n_test_case_block)
    toimage(train_image, mode='L').save('train_image.png')

    # check validation loader
    val_mat = './affNIST/validation.mat'
    aff_mnist_val_loader = AffNistLoader(val_mat, one_hot=True)

    # get validation data
    val_batch_x, val_batch_y = aff_mnist_val_loader.next_batch(n_test_case)

    # reshape input
    val_batch_x = np.reshape(val_batch_x, (-1, 40, 40, 1))

    val_image = form_image(val_batch_x, n_test_case_block)
    toimage(val_image, mode='L').save('val_image.png')

    # check test loader
    test_mat = './affNIST/test.mat'
    aff_mnist_test_loader = AffNistLoader(test_mat, one_hot=True)

    # get test data
    test_batch_x, test_batch_y = aff_mnist_test_loader.next_batch(n_test_case)

    # reshape input
    test_batch_x = np.reshape(test_batch_x, (-1, 40, 40, 1))

    test_image = form_image(test_batch_x, n_test_case_block)
    toimage(test_image, mode='L').save('test_image.png')
