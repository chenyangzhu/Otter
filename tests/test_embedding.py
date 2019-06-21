import numpy as np
from otter.dam.structure import Variable
from otter.dam.graph import Graph
from otter.layers.language import Embedding

with Graph() as g:
    vocab_size = 100
    embed_size = 10
    max_len = 150
    data_len = 1000

    emb = Embedding(max_len, vocab_size, embed_size)

    x = Variable(np.random.randint(0, vocab_size-1, (data_len, max_len)))

    embedded = emb.forward(x)

    print(embedded)