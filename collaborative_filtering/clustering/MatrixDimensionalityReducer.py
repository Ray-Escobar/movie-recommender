import numpy as np

from collaborative_filtering.clustering.AgglomerativeClusterer import AgglomerativeClusterer


class MatrixDimensionalityReducer:
    """
    Reduces the dimensionality of the provided matrix using agglomerative clustering.
    The matrix is first reduced by rows, and then by columns.
    The algorithm assumes that after reducing by rows, the similarity among columns remains approximately the same
    """
    def __init__(self, new_dimension: np.array):
        self.new_dimension = new_dimension


    def reduce_matrix(self, matrix: np.array, row_similarity_matrix: np.array, column_similarity_matrix: np.array):
        """
        Reduces the dimensionality of the matrix using agglomerative clustering.

        :return: The reduces matrix, a dictionary mapping the row space to the reduced row space, a dictionary mapping column space to a reduced column space
        """
        return self.__reduce_matrix(matrix, row_similarity_matrix, column_similarity_matrix, self.new_dimension[0], self.new_dimension[1])


    def __reduce_matrix(self, matrix: np.array, row_sim_mat: np.array, col_sim_mat: np.array, new_num_rows: int, new_num_cols: int):
        mat_reduced_by_row, row_index_to_partition = self.__reduce_by_row(matrix, row_sim_mat, new_num_rows)
        mat_reduced_by_col, col_index_to_partition = self.__reduce_by_row(mat_reduced_by_row.T, col_sim_mat, new_num_cols)

        mat_reduced = mat_reduced_by_col.T

        return mat_reduced, row_index_to_partition, col_index_to_partition





    def __reduce_by_row(self, matrix: np.array, row_sim_mat: np.array, new_num_rows: int):
        clusterer: AgglomerativeClusterer = AgglomerativeClusterer(
            item_indices=[i for i in range(matrix.shape[0])],
            k_clusters=new_num_rows,
            sim_matrix=row_sim_mat
        )

        partitioned_indices = clusterer.get_partitioned_item_indices()

        # compute index to partition dictionary (index_to_partition[index] returns the partition correspondint to the index after the dimensionality reducer was used)
        index_to_partition = dict()

        for partition_index, partition in enumerate(partitioned_indices):
            for index in partition:
                index_to_partition[index] = partition_index

        # compute the matrix reduced by row

        new_matrix = np.zeros(shape=(new_num_rows, matrix.shape[1]))

        for partition_index, partition in enumerate(partitioned_indices):


            zeroless_mean = lambda vec: np.mean(vec[vec > 0]) if len(vec[vec > 0]) > 0 else 0

            avg_row = np.apply_along_axis(zeroless_mean, 0, matrix[partition])
            new_matrix[partition_index] = avg_row

        return new_matrix, index_to_partition