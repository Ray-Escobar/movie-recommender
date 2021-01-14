from collaborative_filtering.Predictor import Predictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
import heapq
import numpy as np

from collaborative_filtering.Utils import get_k_most_similar_neighbors


class GlobalBaselineCollaborativeFilteringPredictor(Predictor):
    def __init__(self, data_matrix, row_similarity_matrix, k_neighbors):
        self.data_matrix = data_matrix
        self.k_neighbors = k_neighbors
        self.row_similarity_matrix = row_similarity_matrix

        self.global_mean: float = self.__compute_global_mean(self.data_matrix)
        self.row_means: np.ndarray = self.__compute_row_means(self.data_matrix)
        self.column_means: np.ndarray = self.__compute_column_means(self.data_matrix)



    def __compute_global_mean(self, data_matrix) -> float:
        return float(np.mean(data_matrix))


    def __compute_row_means(self, data_matrix) -> np.ndarray:
        zeroless_mean = lambda vec: np.mean(vec[vec > 0]) if len(vec[vec > 0]) > 0 else 0
        return np.apply_along_axis(zeroless_mean, -1, data_matrix)

    def __compute_column_means(self, data_matrix) -> np.ndarray:
        zeroless_mean = lambda vec: np.mean(vec[vec > 0]) if len(vec[vec > 0]) > 0 else 0
        return np.apply_along_axis(zeroless_mean, 0, data_matrix)

    def __compute_baseline_estimate(self, row: int, col: int, global_mean: float, row_means: np.array, col_means: np.array):
        return global_mean + (row_means[row] - global_mean) + (col_means[col] - global_mean)


    def predict(self, row: int, col: int):
        ratings, weights, row_indices = get_k_most_similar_neighbors(row, col, self.data_matrix, self.row_similarity_matrix,
                                                        self.k_neighbors)

        target_baseline_estimate = self.__compute_baseline_estimate(row, col, self.global_mean, self.row_means,
                                                             self.column_means)

        # if no collaborative filtering can be performed, simply return the baseline estimate
        if len(ratings) == 0 or np.sum(weights) == 0:
            return target_baseline_estimate

        neighbor_baseline_estimates = np.array([self.__compute_baseline_estimate(neighbor_row, col, self.global_mean,
                                                                         self.row_means, self.column_means) for neighbor_row in row_indices])

        neighbor_deviations = np.array(ratings) - neighbor_baseline_estimates

        # apply pearson similarity on the deviations and add it to the baseline estimate for rating



        return target_baseline_estimate + np.average(neighbor_deviations, weights=weights)



