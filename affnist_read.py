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
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class AffNISTLoader(object):
    def __init__(self, base_dir, how_many=5, one_hot=True):
        if how_many <= 0 or how_many > 32:
            raise ValueError('There are 1 ~ 32 set of training sets, each contains 50,000 training examples.')

        single_train_size = 50000
        self.n_sets = how_many
        self.one_hot = one_hot
        self.n_train_examples = self.n_sets * single_train_size
        self.n_val_examples = self.n_sets * single_val_size
        self.n_test_examples = self.n_sets * single_test_size

        # set relative pathes
        training_dir = os.path.join(base_dir, 'training_batches')
        validation_dir = os.path.join(base_dir, 'validation_batches')
        test_dir = os.path.join(base_dir, 'test_batches')

        # get matlab *.mat files
        training_batches = glob.glob(os.path.join(training_dir, '*.mat'))
        validation_batches = glob.glob(os.path.join(validation_dir, '*.mat'))
        test_batches = glob.glob(os.path.join(test_dir, '*.mat'))

        # sort for convinience
        training_batches.sort(key=natural_keys)
        validation_batches.sort(key=natural_keys)
        test_batches.sort(key=natural_keys)

        # merge needed training examples
        self.train_x = np.empty((self.n_train_examples, 40*40), dtype=np.float32)
        self.train_y = np.empty((self.n_train_examples, 1), dtype=np.float32)
        if self.one_hot:
            self.train_y = np.empty((self.n_train_examples, 10), dtype=np.float32)

        # iterate all examples and merge
        for ii in range(self.n_sets):
            train_mat_fn = training_batches[ii]
            val_mat_fn = validation_batches[ii]
            test_mat_fn = test_batches[ii]
            train_dataset = loadmat(train_mat_fn)
            val_dataset = loadmat(val_mat_fn)
            test_dataset = loadmat(test_mat_fn)

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
        if self.batch_index + batch_size > self.n_train_examples:
            self.batch_index = 0

        x = self.merged_x[self.batch_index:self.batch_index + batch_size]
        y = self.merged_y[self.batch_index:self.batch_index + batch_size]

        self.batch_index += batch_size
        return x, y


def main():
    affNIST_base_dir = 'D:\\db\\affineMnist\\transformed'
    aff_mnist_loader = AffNISTLoader(affNIST_base_dir, how_many=1, one_hot=True)
    batch_size = 128

    for ii in range(aff_mnist_loader.n_train_examples // batch_size):
        # get training data
        batch_x, batch_y = aff_mnist_loader.get_next_batches(batch_size)

        # reshape input
        batch_x = np.reshape(batch_x, (-1, 40, 40, 1))


    # training_dir = os.path.join(affNIST_base_dir, 'training_batches')
    # validation_dir = os.path.join(affNIST_base_dir, 'validation_batches')
    # test_dir = os.path.join(affNIST_base_dir, 'test_batches')
    #
    # training_batches = glob.glob(os.path.join(training_dir, '*.mat'))
    # validation_batches = glob.glob(os.path.join(validation_dir, '*.mat'))
    # test_batches = glob.glob(os.path.join(test_dir, '*.mat'))
    #
    # training_batches.sort(key=natural_keys)
    # validation_batches.sort(key=natural_keys)
    # test_batches.sort(key=natural_keys)
    #
    # for mat_fn in training_batches:
    #     dataset = loadmat(mat_fn)
    #
    #     ans_set = dataset['affNISTdata']['label_one_of_n'].transpose()
    #     train_set = dataset['affNISTdata']['image'].transpose()
    #
    #     print('{:s}: number of examples - {:d}'.format(mat_fn, train_set.shape[0]))

    return


if __name__ == '__main__':
    main()
