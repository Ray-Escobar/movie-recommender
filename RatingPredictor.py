from typing import List

from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
from data_handling.DiskPersistor import DiskPersistor
import numpy as np


class RatingPredictor:

    """
    A class representing a full rating predictor that can be used to predict movie ratings.
    The Rating Predictor can be configured to use one or more prediction strategies
    """
    def __init__(self, data_loader: DataLoader, disk_persistor: DiskPersistor, persistence_id: str, prediction_strategies: List[PredictionStrategy]):
        self.prediction_strategies: List[PredictionStrategy] = prediction_strategies


        self.precomputations_performed = False
        self.disk_persistor = disk_persistor
        self.data_loader = data_loader
        self.persistence_id = persistence_id

        # add the data loader and the disk persistence to all of the prediction strategies
        for strategy in self.prediction_strategies:
            strategy.add_data_loader(data_loader)




    def perform_precomputations(self, force_update: bool = False):
        """
        Performs the precomputations necessary to make the predictions.
        """

        # configure the disk persistor:
        for index, strategy in enumerate(self.prediction_strategies):
            strategy.add_disk_persistor(self.disk_persistor, '{}_{}'.format(self.persistence_id, index), force_update)

        # make the necessary precomputations
        for strategy in self.prediction_strategies:
            strategy.perform_precomputations()

        self.precomputations_performed = True


    def make_average_prediction(self, weights: List[float]):
        """
        Returns the prediction corresponding to taking the average of the predicted values obtained by applying all strategies.
        :return: the average prediction of all the strategies
        """
        weights = np.array(weights)

        if len(self.prediction_strategies) == 0:
            raise Exception('Cannot perform predictions! No strategies are available!')

        if self.precomputations_performed is False:
            raise Exception('The necessary precomputations need to be performed before making predictions!')

        predictions = []

        for strategy in self.prediction_strategies:
            predictions.append(strategy.predict())



        avg_prediction = dict()
        prediction_keys = self.data_loader.get_prediction_instances()


        for prediction_key in prediction_keys:
            prediction_key = tuple(prediction_key)
            ratings = np.array([prediction[prediction_key] for prediction in predictions])


            # filter out the ratings that are 0, since the prediction rating is 0 when the strategy was not able to predict the rating

            indexes = np.where(ratings > 0.0)

            # if none of the strategies can predict the rating, then there is nothing we can do
            if len(indexes) == 0:
                avg_prediction[prediction_key] = 0.0

            ratings_non_zero = ratings[indexes]
            weights_non_zero = weights[indexes]

            avg_prediction[prediction_key] = np.average(ratings_non_zero, weights=weights_non_zero)



        return avg_prediction

