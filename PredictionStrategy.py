import numpy as np

from DataLoader import DataLoader


class PredictionStrategy:
    """
    Abstract class representing a strategy used to predict user ratings.
    Different  prediction strategies extend this class.
    """

    __data_loader_added = False

    def add_data_loader(self, data_loader: DataLoader):
        """
        Configures a data loader to be used by the prediction strategy. The data loader provides the strategy
        the necessary data to make a prediction.
        :param data_loader: the data loader to be added
        :return: the method doesn return anything
        """

        self.__data_loader_added = True
        self.data_loader:DataLoader = data_loader

        # initializes the user-movie instances to be predicted
        self.user_movie_instances_to_be_predicted = self.data_loader.get_prediction_instances()



    def predict(self):
        """
        Makes a user rating prediction based on the data provided by the data loader.
        The particular strategy depends on the implemented strategy.
        :raises: raises an exception if the data loader was not added before being called.
        :return: the predicted ratings, returned as a dictonory indexed by userId and movieId
        """

        if self.__data_loader_added is False:
            raise Exception('You need to add a dataloader before you can make predictions!')



