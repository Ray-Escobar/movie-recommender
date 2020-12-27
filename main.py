import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
from CosineLshUserCollaborativeFiltering import CosineLshUserCollaborativeFiltering
from DataLoader import DataLoader
from DataPathProvider import DataPathProvider
from DiskPersistor import DiskPersistor
from FormulaFactory import SimilarityMeasureType, FormulaFactory
from LocalFileCsvProvider import LocalFileCsvProvider
from NaiveUserCollaborativeFiltering import NaiveUserCollaborativeFiltering
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

# Read the data using pandas
# movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
# users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
# ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
# predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])



# movies_data = data_loader.get_movies_data()
# users_data = data_loader.get_users_data()
# ratings_data = data_loader.get_ratings_data()
# predictions_data = data_loader.get_predictions_data()
# ratings_table = data_loader.get_ratings_map()
#
#print(ratings_matrix)

formula_factory = FormulaFactory()
prediction_strategy: PredictionStrategy = NaiveUserCollaborativeFiltering(3, SimilarityMeasureType.MEANLESS_COSINE_SIMILARITY, formula_factory)
prediction_strategy: PredictionStrategy = CosineLshUserCollaborativeFiltering(5, 6, 1, formula_factory, 3)
prediction_strategy.add_data_loader(data_loader)
prediction_strategy.add_disk_persistor(disk_persistor=DiskPersistor(), persistence_id='lsh', force_update=True)
prediction_strategy.perform_precomputations()
print(prediction_strategy.predict())



# def predict(movies, users, ratings, predictions):
#     number_predictions = len(predictions)
#
#     return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]
#
#
# #####
# ##
# ## SAVE RESULTS
# ##
# #####
#
# ## //!!\\ TO CHANGE by your prediction function
# predictions = predict(movies_description, users_description, ratings_description, predictions_description)
#
# # Save predictions, should be in the form 'list of tuples' or 'list of lists'
# with open(submission_file, 'w') as submission_writer:
#     # Formates data
#     predictions = [map(str, row) for row in predictions]
#     predictions = [','.join(row) for row in predictions]
#     predictions = 'Id,Rating\n' + '\n'.join(predictions)
#
#     # Writes it dowmn
#     submission_writer.write(predictions)