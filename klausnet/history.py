class history_recorder:
    def __init__(self, recorders_list):
        self.recorders_list = recorders_list

    def record(self):
        pass


class Metrics:
    def __init__(self):
        pass

    def record(self):
        pass

# TODO

class categorical_accuracy(Metrics):
    def __init__(self):
        super().__init__()


class sparse_categorical_accuracy(Metrics):
    def __init__(self):
        super().__init__()


class loss(Metrics):
    def __init__(self):
        super().__init__()
