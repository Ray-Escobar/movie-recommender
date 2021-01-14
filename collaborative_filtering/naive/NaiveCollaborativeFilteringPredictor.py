from collaborative_filtering.Predictor import Predictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
import heapq
import numpy as np
from collaborative_filtering.Utils import get_k_most_similar_neighbors


class NaiveCollaborativeFilteringPredictor(Predictor):
    def __init__(self, data_matrix, row_similarity_matrix, k_neighbors):
        self.data_matrix = data_matrix
        self.k_neighbors = k_neighbors
        self.row_similarity_matrix = row_similarity_matrix

    def predict(self, row: int, col: int):
        ratings, weights, _ = get_k_most_similar_neighbors(row, col, self.data_matrix, self.row_similarity_matrix,
                                                        self.k_neighbors)

        if len(ratings) == 0 or np.sum(weights) == 0:
            return 0

        return np.average(ratings, weights=weights)

