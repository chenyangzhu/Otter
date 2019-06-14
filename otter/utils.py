from sklearn.preprocessing import OneHotEncoder


def int2onehot(vocab_size, x):
    one_hot_encoder = OneHotEncoder(categories=[range(vocab_size)]).fit(x)
    one_hot_x = one_hot_encoder.transform(x).toarray()

    return one_hot_x