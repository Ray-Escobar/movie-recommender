from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
import numpy as np
import heapq

# Not yet finished! I do not know if SVD can really be used in this case!!
class SvdLatentFactors(PredictionStrategy):
    """
    Makes predictions by performing an SVD decomposition on the provided matrix
    """

    def __init__(self, r_singular_values):
        self.r_singular_values = r_singular_values

    def add_data_loader(self, data_loader: DataLoader):
        self.ratings_matrix = data_loader.get_ratings_matrix()

    def predict(self):
        return self.__perform_svd_decomposition()


    def __perform_svd_decomposition(self) -> (np.array, np.array, np.array):
        """
        Performs SVD decomposition on the ratings matrix.
        Returns the 3 matrices, U, D and V corresponding to the decomposition
        :return:
        """

        svd = np.linalg.svd(self.ratings_matrix, full_matrices=False)
        sigma = self.__create_diagonal_matrix(svd[1])
        return svd[0], sigma, svd[2]

    def __create_diagonal_matrix(self, singular_values: np.array) -> np.array:
        """
        Creates a diagonal matrix of the r highest singular values. if r > num singular values, all singular values are kept
        :param singular_values: the list of singular values to be used
        :return: a diagonal matrix containing the r largest singular values
        """
        num_singular_values_to_be_kept = min(self.r_singular_values, np.size(singular_values))
        mat_sz = np.size(singular_values)
        mat = np.zeros((mat_sz, mat_sz))

        # find largest singular values
        values = []

        for i, val in enumerate(singular_values.tolist()):
            heapq.heappush(values, (-val, i))

        singular_values_to_be_kept = []

        for _ in range(num_singular_values_to_be_kept):
            singular_values_to_be_kept.append(heapq.heappop(values)[1])

        for i in range(mat_sz):
            if i in singular_values_to_be_kept:
                mat[i][i] = singular_values[i]
            else:
                mat[i][i] = 0

        return mat
