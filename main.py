# -*- coding: utf-8 -*-
from typing import List

from commons.RatingPredictor import RatingPredictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.ClusterCollaborativeFiltering import ClusterCollaborativeFiltering
from collaborative_filtering.global_baseline.ItemGlobalBaselineCollaborativeFiltering import \
    ItemGlobalBaselineCollaborativeFiltering
from collaborative_filtering.global_baseline.UserGlobalBaselineCollaborativeFiltering import \
    UserGlobalBaselineCollaborativeFiltering
from collaborative_filtering.naive.ItemNaiveCollaborativeFiltering import ItemNaiveCollaborativeFiltering
from collaborative_filtering.naive.UserNaiveCollaborativeFiltering import UserNaiveCollaborativeFiltering
from data_handling.DataLoader import DataLoader
from data_handling.DataPathProvider import DataPathProvider
from data_handling.DiskPersistor import DiskPersistor
from commons.FormulaFactory import FormulaFactory
from data_handling.LocalFileCsvProvider import LocalFileCsvProvider

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
submission_file = 'data/submissions/submission.csv'

# Create a data path provider
data_path_provider = DataPathProvider(movies_path=movies_file, users_path=users_file, ratings_path=ratings_file, predictions_path=predictions_file, submission_path=submission_file)

# Creata a data loader
data_loader = DataLoader(data_path_provider=data_path_provider, csv_provider=LocalFileCsvProvider())
disk_persistor = DiskPersistor()
formula_factory = FormulaFactory()

# Create the user similarity matrix matrix if not already created
sym_matrix_results = disk_persistor.perist_computation([
    (lambda: RowPearsonSimilarityMatrix(data_loader.get_ratings_matrix()), 'global_pearson_similarity_matrix'),
    (lambda: RowPearsonSimilarityMatrix(data_loader.get_ratings_matrix().T), 'global_pearson_similarity_matrix_movie')
], force_update=False)

global_pearson_similarity_matrix_user = sym_matrix_results[0]
global_pearson_similarity_matrix_movie = sym_matrix_results[1]



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


def predict_and_write_to_file(predictor: RatingPredictor, force_update: bool, weights: List[float], submission_file: str):


    ## //!!\\ TO CHANGE by your prediction function
    predictions = predict(predictor, force_update=force_update, weights=weights)

    # Save predictions, should be in the form 'list of tuples' or 'list of lists'
    with open(submission_file, 'w') as submission_writer:
        # Formates data
        predictions = [map(str, row) for row in predictions]
        predictions = [','.join(row) for row in predictions]
        predictions = 'Id,Rating\n' + '\n'.join(predictions)

        # Writes it dowmn
        submission_writer.write(predictions)





# User - user and item - item collaborative filtering

naive_colab_predictor: RatingPredictor = RatingPredictor(
    data_loader=data_loader,
    disk_persistor=disk_persistor,
    persistence_id='predictor_naive',
    prediction_strategies=[
        ItemNaiveCollaborativeFiltering(
            k_neighbors=30,
            sim_matrix=global_pearson_similarity_matrix_movie
        ),
        UserNaiveCollaborativeFiltering(
            k_neighbors=30,
            sim_matrix=global_pearson_similarity_matrix_user
        )
    ]
)

predict_and_write_to_file(naive_colab_predictor, True, [0.7, 0.3], 'data/submissions/naive_colab.csv')


# Clustering Collaborative Filtering
clustering_predictor: RatingPredictor = RatingPredictor(
    data_loader=data_loader,
    disk_persistor=disk_persistor,
    persistence_id='predictor_clustering',
    prediction_strategies=[
        ClusterCollaborativeFiltering(
            row_similarity_matrix=global_pearson_similarity_matrix_user,
            col_similarity_matrix=global_pearson_similarity_matrix_movie,
            new_dim_ratio=(0.8, 0.8),
            k_neighbors=35,
            randomized=True,
            randomized_num_extractions=1000,
            random_seed=3
        )
    ]
)

predict_and_write_to_file(clustering_predictor, True, [1], 'data/submissions/clustering.csv')


# User - user and item - item baseline collaborative filtering

global_baseline_predictor: RatingPredictor = RatingPredictor(
    data_loader=data_loader,
    disk_persistor=disk_persistor,
    persistence_id='predictor_baseline',
    prediction_strategies=[
        ItemGlobalBaselineCollaborativeFiltering(
            k_neighbors=30,
            sim_matrix=global_pearson_similarity_matrix_movie
        ),
        UserGlobalBaselineCollaborativeFiltering(
            k_neighbors=30,
            sim_matrix=global_pearson_similarity_matrix_user
        )
    ]
)

predict_and_write_to_file(global_baseline_predictor, True, [0.7, 0.3], 'data/submissions/global_baselines.csv')
