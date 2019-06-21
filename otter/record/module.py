from otter.model import Model


class Saver:
    def __init__(self, model, save_list):
        """ This module saves some features of this model

        args:
            model:      the Model class
            save_list:  a list containing all the metrics we want to save.

        """
        self.model = model
        self.save_list = save_list

    def record(self):
        pass