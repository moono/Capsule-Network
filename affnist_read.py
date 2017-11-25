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


class AffNistTrainLoader(object):
    def __init__(self, training_dir, n_sets=5, one_hot=True):
        if n_sets <= 0 or n_sets > 32:
            raise ValueError('There are 1 ~ 32 set of training sets, each contains 50,000 training examples.')

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
        self.n_samples = self.n_sets * single_train_size

        # merge training examples
        self.train_x = np.empty((self.n_samples, im_size * im_size), dtype=np.float32)
        self.train_y = np.empty((self.n_samples, 1), dtype=np.float32)
        if self.one_hot:
            self.train_y = np.empty((self.n_samples, 10), dtype=np.float32)

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
            self.train_x[start:end] = x
            self.train_y[start:end] = y

        # reset batch index
        self.batch_index = 0

        return

    def get_next_batches(self, batch_size):
        if self.batch_index + batch_size > self.n_samples:
            self.batch_index = 0

        x = self.train_x[self.batch_index:self.batch_index + batch_size]
        y = self.train_y[self.batch_index:self.batch_index + batch_size]

        self.batch_index += batch_size
        return x, y


class AffNistLoader(object):
    def __init__(self, mat_file, one_hot=True):
        # set class attributes
        self.one_hot = one_hot

        # load mat file
        dataset = loadmat(mat_file)

        # load image and label
        self.x = dataset['affNISTdata']['image'].transpose().astype(np.float32)
        self.y = dataset['affNISTdata']['label_int'].transpose().astype(np.float32)
        if self.one_hot:
            self.y = dataset['affNISTdata']['label_one_of_n'].transpose()

        # normalize image to 0 ~ 1
        self.x = self.x / 255.0

        # find number of samples
        self.n_samples = self.y.shape[0]

        # reset batch index
        self.batch_index = 0

        return

    def get_next_batches(self, batch_size):
        if self.batch_index + batch_size > self.n_samples:
            self.batch_index = 0

        x = self.x[self.batch_index:self.batch_index + batch_size]
        y = self.y[self.batch_index:self.batch_index + batch_size]

        self.batch_index += batch_size
        return x, y


def main():
    trainig_dir = './affNIST/training_batches'
    aff_mnist_train_loader = AffNistTrainLoader(trainig_dir, n_sets=3, one_hot=True)
    batch_size = 128
    epochs = 2

    for e in range(epochs):
        for ii in range(aff_mnist_train_loader.n_samples // batch_size):
            # get training data
            batch_x, batch_y = aff_mnist_train_loader.get_next_batches(batch_size)

            # reshape input
            batch_x = np.reshape(batch_x, (-1, 40, 40, 1))
            print('')

    val_mat = './affNIST/validation.mat'
    aff_mnist_val_loader = AffNistLoader(val_mat, one_hot=True)
    for ii in range(aff_mnist_val_loader.n_samples // batch_size):
        # get validation data
        val_x, val_y = aff_mnist_val_loader.get_next_batches(batch_size)

        # reshape input
        val_x = np.reshape(val_x, (-1, 40, 40, 1))
        print('')

    test_mat = './affNIST/test.mat'
    aff_mnist_test_loader = AffNistLoader(test_mat, one_hot=True)
    for ii in range(aff_mnist_test_loader.n_samples // batch_size):
        # get validation data
        test_x, test_y = aff_mnist_test_loader.get_next_batches(batch_size)

        # reshape input
        test_x = np.reshape(test_x, (-1, 40, 40, 1))
        print('')
    return


if __name__ == '__main__':
    main()
