import unittest
import numpy as np

from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.ClusteringPredictor import ClusteringPredictor


class MyTestCase(unittest.TestCase):
    def test_something(self):
        rating_matrix = np.array([
            [0, 3, 1, 0, 5, 2, 0, 0, 5],
            [0, 0, 2, 5, 4, 3, 1, 0, 4],
            [5, 5, 1, 4, 3, 5, 2, 1, 3],
            [4, 4, 5, 3, 0, 3, 1, 2, 4]
        ])

        similarity_matrix_row = RowPearsonSimilarityMatrix(rating_matrix).get_matrix()
        similarity_matrix_column = RowPearsonSimilarityMatrix(rating_matrix.T).get_matrix()

        predictor = ClusteringPredictor(rating_matrix, similarity_matrix_row, similarity_matrix_column, (3, 5), 2)

        predictor.perform_precomputations()

        print(predictor.predict(0, 0))
        print(predictor.predict(0, 3))
        print(predictor.predict(0, 6))
        print(predictor.predict(0, 7))
        print(predictor.predict(1, 7))


if __name__ == '__main__':
    unittest.main()
