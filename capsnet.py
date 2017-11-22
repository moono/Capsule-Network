import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import utils

'''
tf.matmul(): matrix multiplication
tf.multiply(): element wise x * y
tf.layers.xxx()'s default kernel_initializer: glorot_uniform_initializer ==> Xavier uniform initializer
'''

class CapsNet(object):
    def __init__(self):
        # start building graph
        tf.reset_default_graph()

        # set class variables
        self.m_plus = 0.9
        self.m_minus = 1.0 - self.m_plus
        self.lmbd = 0.5
        self.reconstruction_scaler = 0.0005
        self.learning_rate = 0.001

        # we are handling MNIST dataset
        self.im_size = 28
        self.y_dim = 10
        self.inputs_x = tf.placeholder(tf.float32, [None, self.im_size, self.im_size, 1], name='inputs_x')
        self.inputs_y = tf.placeholder(tf.float32, [None, self.y_dim], name='inputs_y')

        # build architecture
        n_k = 9
        n_routing = 3

        # first convolution layer: returns [batch_size, 20, 20, 256]
        relu_conv1_out = self.conv_layer(self.inputs_x, n_filter=256, n_k=n_k)

        # primary caps layer: returns [batch_size, 1152, 8]
        primary_caps_out = self.primary_caps_layer(relu_conv1_out, n_dim=8, n_channel=32, n_k=n_k)

        # digit caps layer: returns [batch_size, 10, 16]
        digit_caps_out = self.digit_caps_layer(primary_caps_out, n_dim=16, n_classes=self.y_dim, n_routing=n_routing)

        # compute length of the digit caps layer output(instantiation vector)
        # to represent probability that a capsule's entity(here, digit) exists
        epsilon = 1e-9
        self.iv_length = tf.sqrt(tf.reduce_sum(tf.square(digit_caps_out), axis=2) + epsilon)

        # softmax the iv_length for finding network's final output(0~9 digit selection)
        self.softmax_iv = tf.nn.softmax(self.iv_length)

        # masking layer: returns [batch_size, 10, 16]
        masked = self.masking_layer(digit_caps_out, self.inputs_y)

        # reconstruction layer: returns [batch_size, 784]
        self.reconstructed = self.reconstruction_layer(masked)

        # loss
        self.margin_loss, self.recon_loss, self.total_loss = self.model_loss(self.iv_length, self.inputs_y,
                                                                             self.reconstructed, self.inputs_x)

        # optimizer
        self.train_opt = self.model_opt(self.total_loss)
        return

    def model_loss(self, iv_length, true_label, recon_out, true_inputs):
        # compute margin loss
        maximum_l = tf.square(tf.maximum(0., self.m_plus - iv_length))
        maximum_r = tf.square(tf.maximum(0., iv_length - self.m_minus))
        maximum = true_label * maximum_l + self.lmbd * (1.0 - true_label) * maximum_r
        margin_loss = tf.reduce_mean(tf.reduce_sum(maximum, axis=1))

        # compute reconstruction loss
        reshaped = tf.reshape(true_inputs, shape=(-1, 784))
        reconstruction_loss = tf.reduce_sum(tf.square(recon_out - reshaped))

        # total loss
        total_loss = margin_loss + self.reconstruction_scaler * reconstruction_loss

        return margin_loss, reconstruction_loss, total_loss

    def model_opt(self, total_loss):
        t_vars = tf.trainable_variables()

        beta1 = 0.5
        train_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(total_loss, var_list=t_vars)

        return train_opt

    @staticmethod
    def squash(sj, norm_axis):
        """
        :param sj: vector to squash
        :param norm_axis: axis for squared norm calculation
        :return: squashed vector same size & dimension as input sj
        """
        epsilon = 1e-9
        sj_squared_norm = tf.reduce_sum(tf.square(sj), axis=norm_axis, keep_dims=True)
        scale = sj_squared_norm / (1.0 + sj_squared_norm) / tf.sqrt(sj_squared_norm + epsilon)
        vj = scale * sj
        return vj

    def conv_layer(self, inputs, n_filter, n_k):
        """
        :param inputs: [batch_size, 28, 28, 1]
        :param n_filter: 256
        :param n_k: 9
        :return: [batch_size, 20, 20, 256]
        """
        with tf.variable_scope('Conv_layer'):
            # [batch_size, 28, 28, 1] => [batch_size, 20, 20, 256]
            layer = tf.layers.conv2d(inputs, filters=n_filter, kernel_size=n_k, strides=1, padding='valid')
            layer = tf.nn.relu(layer)
        return layer

    def primary_caps_layer(self, inputs, n_dim, n_channel, n_k):
        """
        :param inputs: [batch_size, 20, 20, 256]
        :param n_dim: 8
        :param n_channel: 32
        :param n_k: 9
        :return: [batch_size, 1152, 8]
        """
        with tf.variable_scope('PrimaryCaps_layer'):
            # [batch_size, 20, 20, 256] => [batch_size, 6, 6, 256]
            layer = tf.layers.conv2d(inputs, filters=n_dim * n_channel, kernel_size=n_k, strides=2, padding='valid')
            layer = tf.nn.relu(layer)
            l_shape = layer.get_shape().as_list()

            # [batch_size, 6, 6, 256] => [batch_size, 6 * 6 * 32, 8] => [batch_size, 1152, 8]
            # there are 1152 (6 * 6 * 32) capsules (8-D)
            layer = tf.reshape(layer, shape=[-1, l_shape[1] * l_shape[2] * n_channel, n_dim])
            layer = self.squash(layer, norm_axis=2)
        return layer

    def digit_caps_layer(self, inputs, n_dim, n_classes, n_routing):
        """
        :param inputs: [batch_size, 1152, 8]
        :param n_dim: 16
        :param n_classes: 10
        :param n_routing: 3
        :return:[batch_size, 10, 16]
        """
        # get inputs shape sizes as int
        inputs_shape = inputs.get_shape().as_list()
        n_capsules_i = inputs_shape[1]
        n_dim_i = inputs_shape[2]

        # set current capsule size and vector size
        n_capsules_j = n_classes
        n_dim_j = n_dim

        # get batch size as Tensor in order to apply tf.tile()
        batch_size = tf.shape(inputs)[0]

        with tf.variable_scope('DigitCaps_layer'):

            # u_hat = W x u
            with tf.variable_scope('transformation'):
                # create transform matrix W: [1, 1152, 10, 8, 16], trainable
                stddev = 0.02
                w = tf.get_variable(name='W', shape=[1, n_capsules_i, n_capsules_j, n_dim_i, n_dim_j], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=stddev))

                # prepare inputs and W for transform matrix multiplication
                # inputs: [batch_size, 1152, 8] => [batch_size, 1152, 1, 1, 8]
                #         => [batch_size, 1152 (n_capsules_i), 10 (n_capsules_j), 1, 8 (n_dim_i)]
                inputs_reshaped = tf.expand_dims(inputs, axis=2)
                inputs_reshaped = tf.expand_dims(inputs_reshaped, axis=2)
                inputs_reshaped_tiled = tf.tile(inputs_reshaped, multiples=[1, 1, n_capsules_j, 1, 1])

                # W: [1, 1152, 10, 8, 16] => [batch_size, 1152, 10, 8, 16]
                w_tiled = tf.tile(w, multiples=[batch_size, 1, 1, 1, 1])

                # compute u_hat
                # tf.matmul() will do matrix multiplication on last two dimension
                # last 2 dims: [1, 8] * [8, 16] => [1, 16]
                # final output u_hat: [batch_size, 1152, 10, 1, 16]
                u_hat = tf.matmul(inputs_reshaped_tiled, w_tiled)

            with tf.variable_scope('routing'):
                # create coupling coefficients b_ij: [batch_size, 1152, 10, 1, 1], not trainable
                b_ij = tf.zeros([batch_size, n_capsules_i, n_capsules_j, 1, 1], dtype=tf.float32)

                for i in range(n_routing):
                    # weighted sum: s_j = sigma_i {c_ij x u_hat}
                    # c_ij: [batch_size, 1152, 10, 1, 1]
                    c_ij = tf.nn.softmax(b_ij, dim=2)

                    # sum(s_j x u_hat):
                    # sum([batch_size, 1152, 10, 1, 1] x [batch_size, 1152, 10, 1, 16]) => [batch_size, 1, 10, 1, 16]
                    s_j = tf.reduce_sum(tf.multiply(c_ij, u_hat), axis=1, keep_dims=True)

                    # squash
                    # v_j: [batch_size, 1, 10, 1, 16]
                    v_j = self.squash(s_j, norm_axis=2)

                    # update b_ij
                    # u_hat x v_j:
                    # [batch_size, 1152, 10, 1, 16] x [batch_size, 1, 10, 1, 16] => [batch_size, 1152, 10, 1, 16]
                    # sum(u_hat x v_j): [batch_size, 1152, 10, 1, 1]
                    # b_ij: [batch_size, 1152, 10, 1, 1]
                    b_ij += tf.reduce_sum(tf.multiply(u_hat, v_j), axis=4, keep_dims=True)

        # reduce dimension: [batch_size, 1, 10, 1, 16] => [batch_size, 10, 16]
        v_j = tf.squeeze(v_j, axis=3)
        v_j = tf.squeeze(v_j, axis=1)
        return v_j

    def masking_layer(self, inputs, class_label):
        """
        :param inputs: [batch_size, 10, 16]
        :param class_label: [batch_size, 10]
        :return: [batch_size, 10, 16]
        """
        with tf.variable_scope('masking'):
            label = tf.expand_dims(class_label, axis=2)
            masked = tf.multiply(inputs, label)

        return masked

    def reconstruction_layer(self, inputs):
        """
        :param inputs: [batch_size, 10, 16]
        :return: [batch_size, 784]
        """
        inputs_shape = inputs.get_shape().as_list()
        with tf.variable_scope('reconstruction'):
            reshaped = tf.reshape(inputs, shape=[-1, inputs_shape[1] * inputs_shape[2]])
            fc1 = tf.layers.dense(reshaped, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)
            fc3 = tf.layers.dense(fc2, units=784, activation=tf.sigmoid)

        return fc3


def train(net, epochs, batch_size, print_every=50):
    # mnist datset loader
    mnist_dir = 'mnist'
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    steps = 0
    margin_losses = []
    recontruction_losses = []
    total_losses = []
    accuracies = []

    start_time = time.time()

    with tf.Session() as sess:
        # reset tensorflow variables
        sess.run(tf.global_variables_initializer())

        # start training
        for e in range(epochs):
            for ii in range(mnist.train.num_examples // batch_size):
                # get training data
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # reshape input
                batch_x = np.reshape(batch_x, (-1, 28, 28, 1))

                fd = {
                    net.inputs_x: batch_x,
                    net.inputs_y: batch_y
                }

                # Run optimizers
                _ = sess.run(net.train_opt, feed_dict=fd)

                # evaluate losses
                if steps % print_every == 0:
                    margin_loss = net.margin_loss.eval(fd)
                    recon_loss = net.recon_loss.eval(fd)
                    total_loss = net.total_loss.eval(fd)

                    # compute current accuracy
                    accuracy = validate_accruacy(sess, mnist, net)
                    print("Epoch {}/{}...".format(e + 1, epochs),
                          "Margin Loss: {:.4f}...".format(margin_loss),
                          "Reconstruction Loss: {:.4f}...".format(recon_loss),
                          "Total Loss: {:.4f}...".format(total_loss),
                          "Epoch {}/{}... Accuracy: {:.05f}".format(e + 1, epochs, accuracy))

                    # save losses & accuracies
                    margin_losses.append(margin_loss)
                    recontruction_losses.append(recon_loss)
                    total_losses.append(total_loss)
                    accuracies.append(accuracy)
                steps += 1

            # get reconstructed results
            recon_result_fn = 'recon_{:03d}.png'.format(e+1)
            validate_reconstruction(mnist, net, recon_result_fn)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed: {:05.03f} s'.format(elapsed_time))

    # save losses as image
    margin_losses_fn = 'margin-loss.png'
    recon_losses_fn = 'reconstruction-loss.png'
    total_losses_fn = 'total-loss.png'
    accuracy_fn = 'accuracy.png'
    utils.save_loss(margin_losses, 'Margin-loss', margin_losses_fn)
    utils.save_loss(recontruction_losses, 'Reconstruction-loss', recon_losses_fn)
    utils.save_loss(total_losses, 'Total-loss', total_losses_fn)
    utils.save_loss(accuracies, 'Accuracy', accuracy_fn)

    return


def validate_accruacy(sess, mnist, net):
    # accuracy computation
    correct_prediction = tf.equal(tf.argmax(net.softmax_iv, 1), tf.argmax(net.inputs_y, 1))
    correct_prediction_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    # get test data
    test_x = mnist.test.images
    test_y = mnist.test.labels

    # reshape input
    test_x = np.reshape(test_x, (-1, 28, 28, 1))

    n_test_case = test_x.shape[0]
    batch_size = 100
    n_test = n_test_case // batch_size

    cnt_sum = 0.0
    for i in range(n_test):
        start = i * batch_size
        end = start + batch_size
        batch_x = test_x[start:end]
        batch_y = test_y[start:end]
        batch_cnt = sess.run(correct_prediction_count, feed_dict={net.inputs_x: batch_x, net.inputs_y: batch_y})
        cnt_sum += batch_cnt

    accuracy = cnt_sum / float(n_test_case)
    return accuracy


def validate_reconstruction(mnist, net, fn):
    from scipy.misc import toimage

    # get test data
    test_x = mnist.test.images
    test_y = mnist.test.labels

    # reshape input
    test_x = np.reshape(test_x, (-1, 28, 28, 1))

    n_test_case_block = 5
    n_test_case = n_test_case_block * n_test_case_block

    picked_index = np.random.randint(0, test_x.shape[0], n_test_case)
    selected_x = test_x[picked_index]
    selected_y = test_y[picked_index]

    recon = net.reconstructed.eval(feed_dict={net.inputs_x: selected_x, net.inputs_y: selected_y})
    recon = np.reshape(recon, (-1, 28, 28, 1))

    real_image = utils.form_image(selected_x, n_test_case_block)
    recon_image = utils.form_image(recon, n_test_case_block)

    merged = np.concatenate((real_image, recon_image), axis=1)
    toimage(merged, mode='L').save(fn)

    return


def main():
    epochs = 50
    batch_size = 128
    net = CapsNet()

    train(net, epochs, batch_size)
    return


if __name__ == '__main__':
    main()
