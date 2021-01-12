import numpy as np

from FormulaFactory import FormulaFactory, SimilarityMeasureType
from collaborative_filtering.lsh.LocalitySensitiveHashTable import LocalitySensitiveHashTable


class LshCollaborativeFilteringPredictor:
    def __init__(self, data_matrix: np.array, k_neighbors: int, max_query_distance: int, formula_type: SimilarityMeasureType, lsh_table: LocalitySensitiveHashTable,
                 formula_factory: FormulaFactory):
        """
        Initializes the strategy with the provided parameters.
        :param k_neighbors: the number of neighbors to be used in collaborative filtering
        """

        self.k_neighbors = k_neighbors
        self.formula_factory = formula_factory
        self.max_query_distance = max_query_distance

        self.similarity_measure = self.formula_factory.create_similarity_measure(formula_type)
        self.lsh_table: LocalitySensitiveHashTable = lsh_table
        self.data_matrix = data_matrix



        self.most_similar_neighbors = None


    def predict(self, row: int, column: int):
        neighbors = self.lsh_table.query_neighbors(row, self.k_neighbors, column, self.max_query_distance)

        if len(neighbors) == 0:
            return 0.0



        neighbors_data = self.data_matrix[neighbors, :]
        neighbor_ratings = neighbors_data[:, column]


        row_data = self.data_matrix[row, :]



        weights = np.apply_along_axis(lambda x: self.similarity_measure(row_data, x), axis=1, arr = neighbors_data)



        # filter out the negative similarities
        indices = tuple(np.where(weights > 0))
        weights = weights[indices]
        neighbor_ratings = neighbor_ratings[indices]


        # if there are no neighbors remaining, the prediction is not possible
        if len(neighbor_ratings) == 0:
            return 0.0




        rating = np.average(neighbor_ratings, weights=weights)

        return rating














