import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, Reshape, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.nn import softmax_cross_entropy_with_logits

import config as cfg
import loggers as lg

logger = lg.logger_nnet


def loss_with_action_masking(y_true, y_pred):
    logits = y_true
    labels = tf.where(y_true == 0., 1e-6, y_pred)
    return softmax_cross_entropy_with_logits(labels=labels, logits=logits)


class NeuralNetwork:

    def __init__(self, input_shape, output_shape,
                 reg_const=cfg.REG_CONST, learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM):
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None

    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, y, epochs, verbose, validation_split, batch_size):
        lg.logger_nnet.info('FITTING MODEL, {} EPOCHS'.format(epochs))
        return self.model.fit(X, y, epochs=epochs, verbose=verbose,
                              validation_split=validation_split, batch_size=batch_size)

    def save(self, color, version):
        lg.logger_nnet.info('SAVING MODEL {}{:2d}'.format(color, version))
        self.model.save('models/{}_version{}.h5'.format(color, version))

    def load_model(self, color, version):
        lg.logger_nnet.info('LOADING MODEL {}{:2d}'.format(color, version))
        return load_model('models/{}version{}.h5'.format(color, version))

    def printWeightAverages(self):
        layers = self.model.layers
        msg = 'WEIGHT LAYER {:d}: ABSAV = {:f}, SD ={:f}, ABSMAX ={:f}, ABSMIN ={:f}'
        for i, l in enumerate(layers):
            x = l.get_weights()[0]
            lg.logger_nnet.info(msg.format(i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x))))
        lg.logger_nnet.info('------------------')
        msg = 'BIAS LAYER {:d}: ABSAV = {:f}, SD ={:f}, ABSMAX ={:f}, ABSMIN ={:f}'
        for i, l in enumerate(layers):
            x = l.get_weights()[1]
            lg.logger_nnet.info(msg.format(i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x))))
        lg.logger_nnet.info('******************')

    def viewLayers(self):
        layers = self.model.layers
        for i, l in enumerate(layers):
            x = l.get_weights()
            print('LAYER ' + str(i))
            try:
                weights = x[0]
                s = weights.shape
                fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
                channel = 0
                filter = 0
                for i in range(s[2] * s[3]):
                    sub = fig.add_subplot(s[3], s[2], i + 1)
                    sub.imshow(weights[:, :, channel, filter], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                    channel = (channel + 1) % s[2]
                    filter = (filter + 1) % s[3]
            except:
                try:
                    fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
                    for i in range(len(x)):
                        sub = fig.add_subplot(len(x), 1, i + 1)
                        if i == 0:
                            clim = (0, 2)
                        else:
                            clim = (0, 2)
                        sub.imshow([x[i]], cmap='coolwarm', clim=clim, aspect="auto")
                    plt.show()

                except:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1), aspect="auto")
                        plt.show()
                    except:
                        pass

            plt.show()

        lg.logger_nnet.info('------------------')


class ResidualNN(NeuralNetwork):

    def __init__(self, input_shape=cfg.IN_SHAPE, output_shape=cfg.OUT_SHAPE,
                 reg_const=cfg.REG_CONST, learning_rate=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                 hidden_layers=cfg.HIDDEN_LAYERS):
        super().__init__(input_shape=input_shape, output_shape=output_shape,
                         reg_const=reg_const, learning_rate=learning_rate, momentum=momentum)
        self.hidden_layers = hidden_layers
        self.model = self._build_model()

    def _conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   data_format="channels_last", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def _residual_layer(self, x0, filters, kernel_size):
        x = self._conv_layer(x0, filters, kernel_size)
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   data_format="channels_last", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([x0, x])
        x = LeakyReLU()(x)
        return x

    def _value_head(self, x):
        x = Conv2D(filters=1, kernel_size=1,
                   data_format="channels_last", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(20, use_bias=False, activation='linear',
                  kernel_regularizer=l2(self.reg_const))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias=False, activation='tanh',
                  kernel_regularizer=l2(self.reg_const),
                  name="value_head")(x)

        return x

    def _policy_head(self, x):
        x = Conv2D(filters=2, kernel_size=1,
                   data_format="channels_last", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(np.prod(self.output_shape), use_bias=False, activation='linear',
                  kernel_regularizer=l2(self.reg_const))(x)
        x = Reshape(self.output_shape, name='policy_head')(x)
        return x

    def _build_model(self):
        input_block = Input(shape=self.input_shape, name='input')

        x = self._conv_layer(input_block, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
        for hidden_layer in self.hidden_layers[1:]:
            x = self._residual_layer(x, hidden_layer['filters'], hidden_layer['kernel_size'])

        value_head = self._value_head(x)

        model = Model(inputs=[input_block], outputs=[value_head])
        model.compile(loss={'value_head': 'mean_squared_error'},
                      optimizer=SGD(lr=self.learning_rate, momentum=self.momentum))

        return model

    def state_to_model_input(self, state):
        converted = state.convert_into_cnn()
        model_input = np.reshape(converted, self.input_shape)
        return model_input

    def predict(self, state):
        """
            :return: the value of the state
        """
        input_to_model = np.array([self.state_to_model_input(state)])

        preds = self.model.predict(input_to_model)
        value = preds[0, 0]
        return value

    def map_actions(self, logits, actions):
        """
        :return: an array of the predicted values of the actions

        Understanding the output mapping:
        32 levels in 2 groups of 16,
        every group is one axis: rows, columns (with this order):
            - the first layer of every group represents moving by -8 cells on that axis
            - the last layer of every group represents moving by +8 cells on that axis
        """
        pred = np.empty(len(actions), dtype=float)
        for i, (a_from, a_to) in enumerate(actions):
            distance_x, distance_y = np.subtract(a_to, a_from)
            if distance_x != 0:
                # up or down
                offset = 8
                if distance_x > 0:  # because when x=1 I want the 8th layer as x can't be 0
                    offset = 7
                # so distance_x + offset is my layer
                layer = distance_x + offset
            else:
                # left or right
                offset = 24
                if distance_y > 0:  # because when x=1 I want the 8th layer as x can't be 0
                    offset = 23
                layer = distance_y + offset

            idx = (a_from[0], a_from[1], layer)
            pred[i] = logits[idx]
        return pred

    def map_into_action_space(self, actions, pi):
        actions_space = np.zeros(self.output_shape, dtype=float)
        for i, (a_from, a_to) in enumerate(actions):
            distance_x, distance_y = np.subtract(a_to, a_from)
            if distance_x != 0:
                # up or down
                offset = 8
                if distance_x > 0:  # because when x=1 I want the 8th layer as x can't be 0
                    offset = 7
                # so distance_x + offset is my layer
                layer = distance_x + offset
            else:
                # left or right
                offset = 24
                if distance_y > 0:  # because when x=1 I want the 8th layer as x can't be 0
                    offset = 23
                layer = distance_y + offset

            idx = (a_from[0], a_from[1], layer)
            actions_space[idx] = pi[i]
        return actions_space

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
