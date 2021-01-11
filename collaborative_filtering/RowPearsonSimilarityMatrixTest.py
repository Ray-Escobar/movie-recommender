import unittest
import numpy as np

from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix


class MyTestCase(unittest.TestCase):
    def test_similarity_computation(self):
        rating_matrix = np.array([
            [0, 3, 1, 0, 5, 2, 0, 0, 5],
            [0, 0, 2, 5, 4, 3, 1, 0, 4],
            [5, 5, 1, 4, 3, 5, 2, 1, 3],
            [4, 4, 5, 3, 0, 3, 1, 2, 4]
        ])

        similarity_matrix = RowPearsonSimilarityMatrix(rating_matrix)

        print(similarity_matrix.get_matrix())


if __name__ == '__main__':
    unittest.main()
