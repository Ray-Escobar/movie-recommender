from data_handling.DataLoader import DataLoader
from data_handling.DiskPersistor import DiskPersistor


class PredictionStrategy:
    """
    Abstract class representing a strategy used to predict user ratings.
    Different  prediction strategies extend this class.
    """

    __data_loader_added = False

    __disk_persistor_added = False

    __precomputations_performed = False

    disk_persistor = None

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




    def add_disk_persistor(self, disk_persistor: DiskPersistor, persistence_id: str, force_update = False):
        """
        Configures the disk persistor to be used by the prediction strategy.
        :param disk_persistor: the disk persistor to be used
        :param force_update: if true, the computations for the stored objects will be redone and overwritten to disk
        """
        self.__disk_persistor_added = True
        self.disk_persistor = disk_persistor
        self.force_update = force_update
        self.persistence_id = persistence_id





    def perform_precomputations(self):
        """
        Should be called before the predict method in order to perform the necessary precomputations for the prediction
        """

        if self.__data_loader_added is False:
            raise Exception('You need to add a dataloader before you can make predictions!')

        self.__precomputations_performed = True





    def predict(self):
        """
        Makes a user rating prediction based on the data provided by the data loader.
        The particular strategy depends on the implemented strategy.
        :raises: raises an exception if the data loader was not added before being called.
        :return: the predicted ratings, returned as a dictonory indexed by userId and movieId
        """

        if self.__precomputations_performed is False:
            raise Exception('You need to perform precomputations before making predictions')

        pass



