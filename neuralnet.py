from abc import ABC, abstractmethod

import numpy as np
from keras import Input

from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras.optimizers import SGD
from keras.regularizers import l2

import pytablut.config as cfg


class NeuralNetwork(ABC):

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
        self.model.fit(X, y, epochs=epochs, verbose=verbose,
                       validation_split=validation_split, batch_size=batch_size)

    def save(self, color, version):
        self.model.save('models/{}_version{}.h5'.format(color, version))

    def load_model(self, version):
        return load_model('models/version{}.h5'.format(version))

    @abstractmethod
    def _build_model(self):
        pass


class ResidualNN(NeuralNetwork):

    def __init__(self, reg_const, learning_rate, input_shape, output_shape, hidden_layers):
        super().__init__(reg_const, learning_rate, input_shape, output_shape)
        self.hidden_layers = hidden_layers
        self.model = self._build_model()

    def _conv_layer(self, x, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   data_format="channels_first", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        return x

    def _residual_layer(self, x0, filters, kernel_size):
        x = self._conv_layer(x0, filters, kernel_size)
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   data_format="channels_first", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = add([x0, x])
        x = LeakyReLU()(x)
        return x

    def _value_head(self, x):
        x = Conv2D(filters=1, kernel_size=1,
                   data_format="channels_first", padding="same",
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
                   data_format="channels_first", padding="same",
                   use_bias=False, activation="linear",
                   kernel_regularizer=l2(self.reg_const))(x)
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(self.output_shape, use_bias=False, activation='linear',
                  kernel_regularizer=l2(self.reg_const),
                  name='policy_head')(x)
        return x

    def _build_model(self):
        input_block = Input(shape=self.input_shape, name='input')

        x = self._conv_layer(input_block, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])
        for hidden_layer in self.hidden_layers[1:]:
            x = self._residual_layer(x, hidden_layer['filters'], hidden_layer['kernel_size'])

        value_head = self._value_head(x)
        policy_head = self._policy_head(x)

        model = Model(inputs=[input], outputs=[value_head, policy_head])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': custom},
                      optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})

        return model

    def state_to_model_input(self, state):
        # TODO finish this conversion
        model_input = np.reshape(state, self.input_shape)
        return model_input