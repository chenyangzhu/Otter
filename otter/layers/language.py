import numpy as np
from otter.layers.common import Layer
from otter.dam.structure import Variable
from otter.utils import last_dim_one_hot


class Embedding(Layer):

    """
    From one-hot to embedding
    """

    def __init__(self, embed_size):
        """
        :param embed_size:  embed_size
        """

        super().__init__()

        self.embed_size = embed_size
        self.initialize = True

    def forward(self, x: Variable):
        """
        :param x: x [:::::: vocab_size] a one-hot value
        Thing is, we don't really care what's at the front.
        We only need to use the last dimension ( which must be a one-hot) to find its mapping.
        :return:
        """
        self.vocab_size = x.shape[-1]
        if self.initialize:
            self.mapping = Variable(np.random.normal(0, 1, (self.vocab_size, self.embed_size)),
                                    trainable=True)
            self.initialize = False

        # First, find the corresponding word representation
        embedded_word = x.dot(self.mapping)  # n x embed_size
        return embedded_word

    def test_forward(self, x: Variable):
        return self.forward(x)
#
#
# if __name__ == "__main__":
#     n = 1000
#     vocab_size = 300
#     x = np.random.randint(0, vocab_size, (n, 1)) # n data, 1 word
#
#     one_hot_x = last_dim_one_hot(x, vocab_size)
#
#     train_x = Variable(one_hot_x)
#
#     layer = Embedding(embed_size=1000)
#     output = layer.forward(train_x)
#     print(output.shape)  # n x embed_size