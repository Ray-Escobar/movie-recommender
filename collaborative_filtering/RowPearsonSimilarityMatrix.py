from typing import List
import numpy as np

from FormulaFactory import FormulaFactory, SimilarityMeasureType


class RowPearsonSimilarityMatrix:
    """
    Represents a Perason similarity matrix computed based on the rows of the provided matrix,
    in an efficient way as described in this article.
    """

    def __init__(self, data_matrix: List[List]):
        self.data_matrix = data_matrix
        self.sim_matrix = self.compute_similarity_matrix()

    def get_similarity(self, user_index_1: int, user_index_2: int):
        return self.sim_matrix[user_index_1, user_index_2]

    def get_matrix(self):
        return self.sim_matrix

    def compute_similarity_matrix(self):
        # for each row, pre-compute it's mean and meanless average

        print("Generating Similarity Matrix...")

        print("Performing precomputations...")

        vectorized_data: np.array = np.array(self.data_matrix).astype('float64')
        means: np.array = np.apply_along_axis(lambda row: np.mean(row[row != 0]), -1, vectorized_data) # means per row

        meanless_vectorized_data: np.array = vectorized_data - means.reshape(means.shape[0], 1)


        meanless_vectorized_data[vectorized_data == 0] = 0 # contains a list of rows in which the average for each row was substrated for non-zero instances

        sqrt_of_sum = lambda x: np.sqrt(np.sum(x))
        meanless_square_averages: np.array = np.apply_along_axis(sqrt_of_sum, -1, np.power(meanless_vectorized_data, 2)) # the squared root of the square sum of the vectors above

        # now we need to compute the similarity matrix between all possible row combinations

        sim_matrix: np.array = np.zeros(shape=(vectorized_data.shape[0], vectorized_data.shape[0])) # the similarity matrix is the number of rows by the number of rows in size

        print("Generating matrix...")
        num_iterations = 0

        for i, _ in enumerate(vectorized_data):
            for j, _ in enumerate(vectorized_data):

                num_iterations += 1
                print('Progress {} / {}'.format(num_iterations, vectorized_data.shape[0] ** 2))

                # the numerator
                numerator = np.sum(meanless_vectorized_data[i] * meanless_vectorized_data[j])

                # the denominator
                denominator = meanless_square_averages[i] * meanless_square_averages[j]



                pearson_similarity = 0

                if denominator != 0:
                    pearson_similarity = numerator / denominator

                sim_matrix[i, j] = pearson_similarity



        print("Similarity Matrix Generation Finished")

        return sim_matrix



