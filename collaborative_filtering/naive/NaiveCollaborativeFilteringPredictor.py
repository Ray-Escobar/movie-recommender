from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
import heapq
import numpy as np


class NaiveCollaborativeFilteringPredictor:
    def __init__(self, data_matrix, row_similarity_matrix, k_neighbors):
        self.data_matrix = data_matrix
        self.k_neighbors = k_neighbors
        self.row_similarity_matrix = row_similarity_matrix

    def predict(self, row: int, col: int):
        # get the k most similar rows that have a value at that particular column
        most_similar_neighbors = []

        for i, row_vec in enumerate(self.data_matrix):
            if row_vec[col] == 0 or i == row:
                continue

            sim = self.row_similarity_matrix.get_similarity(row, i)

            if sim < 0:
                continue

            heapq.heappush(most_similar_neighbors, (sim, i))

            if len(most_similar_neighbors) > self.k_neighbors:
                heapq.heappop(most_similar_neighbors)

        # if no similar neighbors that have a rating for the given column are found, return a rating of 0
        if len(most_similar_neighbors) == 0:
            return 0

        ratings = [self.data_matrix[neighbor[1]][col] for neighbor in most_similar_neighbors]
        weights = [neighbor[0] for neighbor in most_similar_neighbors]

        if np.sum(weights) == 0:
            return 0

        return np.average(ratings, weights=weights)

