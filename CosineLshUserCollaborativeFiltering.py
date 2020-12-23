from PredictionStrategy import PredictionStrategy
import numpy as np

class CosineLshUserCollaborativeFiltering(PredictionStrategy):
    """
    Makes predictions based on a more efficient collaborative filtering that uses and lsh to query for
    the closest users in terms of cosine distances.
    """

    def __init__(self, k_neighbors):
        """
        Initializes the strategy with the provided parameters.
        :param k_neighbors: the number of neighbors to be used in collaborative filtering
        """
        self.k_neighbors = k_neighbors


    def predict(self):
        """
        Makes predictions based on user-user collaborative filtering.
        """
        pass


class LSH:
    """
    LSH data structure that can be used to query the k nearest neighbors
    """
    def __init__(self, ratings_matrix: np.array, user_id_vector: np.array, movie_id_vector: np.array, signiture_length: int):
        """
        Initializes a new LSH of the provided ratings_matrix.

        :param ratings_matrix:  the ratings matrix create the LSH for
        :param user_id_vector: a vector specifying the user id corresponding to each row
        :param movie_id_vector: a vector specifying the movie id for each column
        :param signiture_length: specifies the length of the signitures to be computed
        """
        self.ratings_matrix = ratings_matrix
        self.user_id_vector = user_id_vector
        self.movie_id_vector = movie_id_vector
        self.bins = dict()
        self.signiture_length = signiture_length

    def __compute_signiture_matrix(self):
        """
        Computes the signiture matrix the data. The matrix will have num_movies columns and signiture_length rows, each column representing
        the signiture of one movie. The following method will be used to compute the signitures:
        https://bogotobogo.com/Algorithms/Locality_Sensitive_Hashing_LSH_using_Cosine_Distance_Similarity.php

        :return: the signiture matrix computed based on the data inside the lsh object
        """