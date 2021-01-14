from typing import List, Tuple

from collaborative_filtering.Predictor import Predictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
import heapq
import numpy as np


def predict_instances_based_on_predictor(predictor: Predictor, instances_to_be_predicted: List[Tuple[int, int]],
                                         user_id_to_row: dict, movie_id_to_col: dict, transpose: bool = False) -> dict:
    """
    Makes a series of predictions using the provided predictor.

    :param predictor: - the predictor used to make the predictions
    :param instances_to_be_predicted: - a list of tuples containing the user id and movie id to predict ratings for
    :param user_id_to_row: - a dictionary converting the user id to a row in the original ratings matrix
    :param movie_id_to_col: - a dictionaru converting the movie id to a column in the original ratings matrix
    :param transpose: - true, if the transpose of the ratings matrix should be used (like in the case of item collaborative filtering), false otherwise
    :return: a dictionary indexed by (user_id, movie_id), containing the predicted ratings
    """

    predictions = dict()

    print("Starting predictions...")

    predictions_num = len(instances_to_be_predicted)
    num_prediction = 0

    for user_id, movie_id in instances_to_be_predicted:
        num_prediction += 1
        print('Progress {} / {}'.format(num_prediction, predictions_num))

        if not transpose:
            row = user_id_to_row[user_id]
            column = movie_id_to_col[movie_id]
        else:
            row = movie_id_to_col[movie_id]
            column = user_id_to_row[user_id]

        rating = predictor.predict(row, column)

        predictions[(user_id, movie_id)] = rating

    print("Finished predictions!")

    return predictions


def get_k_most_similar_neighbors(row: int, col: int, data_matrix, row_similarity_matrix: RowPearsonSimilarityMatrix,
                                 k_neighbors) -> (List, List, List):
    """
    Returns k most similar row-wise neighbors from the provided data.

    :param row: the row corresponding to the prediction
    :param col: the columns corresponding to the prediction
    :param data_matrix: the data matrix containing the ratings
    :param similarity_matrix: the matrix encoding the similarities between all rows
    :param k_neighbors: the number of neighbors to search for
    :return: 3 arrays, the first one containing the similarity for each neighbor, and the second one containing
    the actual rating of the neighbors on column col, the 3rd one containing the rows corresponding to each neighbor
    """

    # get the k most similar rows that have a value at that particular column
    most_similar_neighbors = []

    for i, row_vec in enumerate(data_matrix):
        if row_vec[col] == 0 or i == row:
            continue

        sim = row_similarity_matrix.get_similarity(row, i)

        if sim < 0:
            continue

        heapq.heappush(most_similar_neighbors, (sim, i))

        if len(most_similar_neighbors) > k_neighbors:
            heapq.heappop(most_similar_neighbors)

    # if no similar neighbors that have a rating for the given column are found, return a rating of 0
    if len(most_similar_neighbors) == 0:
        return np.array([]), np.array([]), np.array([])

    ratings = [data_matrix[neighbor[1]][col] for neighbor in most_similar_neighbors]
    weights = [neighbor[0] for neighbor in most_similar_neighbors]
    row_indices = [neighbor[1] for neighbor in most_similar_neighbors]

    return ratings, weights, row_indices
