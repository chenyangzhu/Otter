import numpy as np
from otter.layers.common import Layer
from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.activation import softmax
from otter.loss import sparse_categorical_crossentropy
from otter.layers.language import Embedding

from sklearn.preprocessing import OneHotEncoder

with Graph() as g:
    vocab_size = 100
    embed_size = 10
    max_len = 150
    data_len = 1000

    emb = Embedding(max_len, vocab_size, embed_size)

    x = Variable(np.random.randint(0, vocab_size-1, (data_len, max_len)))

    embedded = emb.train_forward(x)

    print(embedded)