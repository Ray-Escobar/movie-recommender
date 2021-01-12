import unittest

import numpy as np
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.AgglomerativeClusterer import AgglomerativeClusterer
from collaborative_filtering.clustering.MatrixDimensionalityReducer import MatrixDimensionalityReducer

class MyTestCase(unittest.TestCase):
    def test_dim_reducer(self):
        rating_matrix = np.array([
            [0, 3, 1, 0, 5, 2, 0, 0, 5],
            [0, 0, 2, 5, 4, 3, 1, 0, 4],
            [5, 5, 1, 4, 3, 5, 2, 1, 3],
            [4, 4, 5, 3, 0, 3, 1, 2, 4]
        ])

        similarity_matrix_row = RowPearsonSimilarityMatrix(rating_matrix).get_matrix()
        similarity_matrix_column = RowPearsonSimilarityMatrix(rating_matrix.T).get_matrix()

        reducer: MatrixDimensionalityReducer = MatrixDimensionalityReducer((2, 3))

        mat, row_dict, col_dict = reducer.reduce_matrix(rating_matrix, similarity_matrix_row, similarity_matrix_column)
        print(mat)
        print(row_dict)
        print(col_dict)




if __name__ == '__main__':
    unittest.main()
