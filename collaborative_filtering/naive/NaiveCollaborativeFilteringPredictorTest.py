import unittest
import numpy as np

from collaborative_filtering.naive.NaiveCollaborativeFilteringPredictor import NaiveCollaborativeFilteringPredictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix


class MyTestCase(unittest.TestCase):
    def testNaiveCollaborativeFilteringPredictor(self):
        rating_matrix = np.array([
            [0, 3, 1, 0, 5, 2, 0, 0, 5],
            [0, 0, 2, 5, 4, 3, 1, 0, 4],
            [5, 5, 1, 4, 3, 5, 2, 1, 3],
            [4, 4, 5, 3, 0, 3, 1, 2, 4]
        ])

        predictor = NaiveCollaborativeFilteringPredictor(data_matrix=rating_matrix, k_neighbors=2, row_similarity_matrix=RowPearsonSimilarityMatrix(data_matrix=rating_matrix))

        print(predictor.predict(0, 0))
        print(predictor.predict(0, 3))
        print(predictor.predict(0, 6))
        print(predictor.predict(0, 7))
        print(predictor.predict(1, 7))






if __name__ == '__main__':
    unittest.main()
