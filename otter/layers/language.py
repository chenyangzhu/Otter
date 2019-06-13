import numpy as np
from otter.layers.common import Layer
from otter.dam.structure import Variable
from otter.utils import int2onehot


class Embedding(Layer):

    """
    From one-hot to embedding
    """

    def __init__(self, input_dim, vocab_size, embed_size):
        """

        :param input_dim:   max_len specified
        :param vocab_size:  vocabulary size
        :param embed_size:  embed_size
        """

        super().__init__()

        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.mapping = Variable(np.random.normal(0, 1, (vocab_size, embed_size)),
                                trainable=True)
        # self.w = Variable(np.random.normal(0, 1, (embed_size, 1)), trainable=True)

    def train_forward(self, x: Variable):
        """
        :param x: x is the index for each word [batch x input_dim]
        :return:
        """

        # TODO
        # print(x.value.shape)
        one_hot_x = Variable(int2onehot(self.vocab_size, x.value))  # n x input_dim x vocab_size

        one_hot_x = np.zeros((x.value.shape[0], self.input_dim, self.vocab_size))
        for i in range(x.value.shape[0]):
            one_hot_x[i] = int2onehot(self.vocab_size, x.value[i])

        # First, find the corresponding word representation
        self.embedded_word = one_hot_x.dot(self.mapping)  # n x embed_size

        return self.embedded_word
