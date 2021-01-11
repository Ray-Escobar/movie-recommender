import unittest
import numpy as np

from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.AgglomerativeClusterer import AgglomerativeClusterer


class MyTestCase(unittest.TestCase):
    def test_clusterer(self):

        rating_matrix = np.array([
            [0, 3, 1, 0, 5, 2, 0, 0, 5],
            [0, 0, 2, 5, 4, 3, 1, 0, 4],
            [5, 5, 1, 4, 3, 5, 2, 1, 3],
            [4, 4, 5, 3, 0, 3, 1, 2, 4]
        ]).T



        similarity_matrix = RowPearsonSimilarityMatrix(rating_matrix)

        print(similarity_matrix.get_matrix())


        clusterer = AgglomerativeClusterer(item_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8], k_clusters=3, sim_matrix=similarity_matrix.get_matrix())

        clusters = clusterer.get_partitioned_item_indices()

        print('Original Matrix')
        print(rating_matrix)

        print(clusters)




if __name__ == '__main__':
    unittest.main()
