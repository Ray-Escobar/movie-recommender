from typing import Tuple

import numpy as np

from collaborative_filtering.naive.NaiveCollaborativeFilteringPredictor import NaiveCollaborativeFilteringPredictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.MatrixDimensionalityReducer import MatrixDimensionalityReducer


class ClusteringPredictor:
    def __init__(self, data_matrix: np.array, row_similarity_matrix: np.array, col_similarity_matrix: np.array, new_dim: Tuple[int, int], k_neighbors: int, randomized: bool = False, randomized_num_extractions: int = 100, random_seed: int = 3):
        self.data_matrix = data_matrix
        self.row_similarity_matrix = row_similarity_matrix
        self.col_similarity_matrix = col_similarity_matrix
        self.mat_reducer = MatrixDimensionalityReducer(new_dim, randomized, randomized_num_extractions, random_seed)
        self.k_neighbors = k_neighbors # the number of neighbors to be used in case the reduced matrix still has a miss



        self.reduced_matrix = None
        self.reduced_row_sim_matrix = None
        self.colab_predictor = None
        self.row_to_reduced_index = None
        self.col_to_reduced_index = None


    def perform_precomputations(self):
        self.reduced_matrix, self.row_to_reduced_index, self.col_to_reduced_index = self.mat_reducer.reduce_matrix(self.data_matrix, self.row_similarity_matrix, self.col_similarity_matrix)

        # compute the row
        self.reduced_row_sim_matrix = RowPearsonSimilarityMatrix(self.reduced_matrix)

        print(self.reduced_matrix)

        self.colab_predictor: NaiveCollaborativeFilteringPredictor = NaiveCollaborativeFilteringPredictor(self.reduced_matrix, self.reduced_row_sim_matrix, self.k_neighbors)

    def predict(self, row: int, col: int):
        if self.reduced_matrix is None or self.reduced_row_sim_matrix is None or self.colab_predictor is None:
            raise Exception('Precomputations were not performed properly')

        reduced_row = self.row_to_reduced_index[row]
        reduced_col = self.col_to_reduced_index[col]

        if self.reduced_matrix[reduced_row][reduced_col] > 0:
            return self.reduced_matrix[reduced_row][reduced_col]

        return self.colab_predictor.predict(reduced_row, reduced_col)

