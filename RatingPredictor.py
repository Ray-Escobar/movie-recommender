from typing import List

from DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
import numpy as np


class RatingPredictor:
    """
    A class representing a full rating predictor that can be used to predict movie ratings.
    The Rating Predictor can be configured to use one or more prediction strategies
    """
    def __init__(self, data_loader: DataLoader, prediction_strategies: List[PredictionStrategy]):
        self.data_loader:DataLoader = data_loader
        self.prediction_strategies:List[PredictionStrategy] = prediction_strategies

        # add the data loader to all of the prediction strategies

        for strategy in self.prediction_strategies:
            strategy.add_data_loader(self.data_loader)


    def make_average_prediction(self):
        """
        Returns the prediction corresponding to taking the average of the predicted values obtained by applying all strategies.
        :return: the average prediction of all the strategies
        """

        return None #to be implemented

