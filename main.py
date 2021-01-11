# -*- coding: utf-8 -*-
from typing import List

import sys
sys.path.append('.')

from RatingPredictor import RatingPredictor
from collaborative_filtering.CosineLshUserCollaborativeFiltering import CosineLshUserCollaborativeFiltering
from collaborative_filtering.ItemLshCollaborativeFiltering import ItemLshCollaborativeFiltering
from collaborative_filtering.UserLshCollaborativeFiltering import UserLshCollaborativeFiltering
from data_handling.DataLoader import DataLoader
from data_handling.DataPathProvider import DataPathProvider
from data_handling.DiskPersistor import DiskPersistor
from FormulaFactory import SimilarityMeasureType, ScoringMeasureType, FormulaFactory
from data_handling.LocalFileCsvProvider import LocalFileCsvProvider
from collaborative_filtering.NaiveUserCollaborativeFiltering import NaiveUserCollaborativeFiltering
from matrix_factorization.UvDecomposition import UvDecomposer
from PredictionStrategy import PredictionStrategy

"""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: Pietro, Catalin
SURNAME: Vigilanza, Lupau
STUDENT ID: -, 5042143
KAGGLE ID: -, C.P.Lupau@student.tudelft.nl


### NOTES
This files is an example of what your code should look like. 
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'

# Create a data path provider
data_path_provider = DataPathProvider(movies_path=movies_file, users_path=users_file, ratings_path=ratings_file, predictions_path=predictions_file, submission_path=submission_file)

# Creata a data loader
data_loader = DataLoader(data_path_provider=data_path_provider, csv_provider=LocalFileCsvProvider())
disk_persistor = DiskPersistor()
formula_factory = FormulaFactory()




# movies_data = data_loader.get_movies_data()
# users_data = data_loader.get_users_data()
# ratings_data = data_loader.get_ratings_data()
# predictions_data = data_loader.get_predictions_data()
# ratings_table = data_loader.get_ratings_map()
#
#print(ratings_matrix)

predictor: RatingPredictor = RatingPredictor(
    data_loader=data_loader,
    disk_persistor=disk_persistor,
    persistence_id='predictor',
    prediction_strategies=[
        UvDecomposer(
            d = 5,
            delta = 1,
            iterations = 100,
            formula_factory=formula_factory,
            scorer_type= ScoringMeasureType.TRUE_RMSE
        )
    ]
)


'''
prediction_strategies=[
        ItemLshCollaborativeFiltering(
            k_neighbors=30,
            signiture_length=20,
            max_query_distance=5000,
            formula_factory=formula_factory,
            random_seed=4,
        ),
        UserLshCollaborativeFiltering(
            k_neighbors=30,
            signiture_length=20,
            max_query_distance=5000,
            formula_factory=formula_factory,
            random_seed=4,
        ),
        UvDecomposer(
            d = 5,
            delta = 1,
            iterations = 100,
            formula_factory=formula_factory
        )
    ]
'''



def predict(predictor: RatingPredictor, force_update: bool, weights: List[float]):
    predictor.perform_precomputations(force_update=force_update)

    predictions = predictor.make_average_prediction(weights=weights).values()
    predictions = list(predictions)
    number_predictions = len(predictions)

    return [[idx, predictions[idx - 1]] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####

## //!!\\ TO CHANGE by your prediction function
#predictions = predict(predictor, False, [0.7, 0.3])
predictions = predict(predictor, False, [1])

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)