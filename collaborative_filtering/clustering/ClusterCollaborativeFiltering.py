from typing import Tuple

from commons.PredictionStrategy import PredictionStrategy

from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.Utils import predict_instances_based_on_predictor
from collaborative_filtering.clustering.ClusteringPredictor import ClusteringPredictor


class ClusterCollaborativeFiltering(PredictionStrategy):
    def __init__(self, row_similarity_matrix: RowPearsonSimilarityMatrix, col_similarity_matrix: RowPearsonSimilarityMatrix, new_dim_ratio: Tuple[float, float], k_neighbors: int, randomized: bool = False, randomized_num_extractions: int = 100, random_seed: int = 3):
        self.row_similarity_matrix = row_similarity_matrix.get_matrix()
        self.col_similarity_matrix = col_similarity_matrix.get_matrix()

        # compute the actual new dimensions
        # num rows
        new_dim_num_rows = int(new_dim_ratio[0] * self.row_similarity_matrix.shape[0])
        new_dim_num_cols = int(new_dim_ratio[1] * self.col_similarity_matrix.shape[0])
        self.new_dim = (new_dim_num_rows, new_dim_num_cols)


        self.k_neighbors = k_neighbors
        self.randomized = randomized
        self.randomized_num_extractions = randomized_num_extractions
        self.random_seed = random_seed

    def perform_precomputations(self):
        PredictionStrategy.perform_precomputations(self)

        self.ratings_matrix = self.data_loader.get_ratings_matrix()

        self.user_id_vector, self.movie_id_vector = self.data_loader.get_rating_matrix_user_and_movie_data()
        self.user_id_to_row_dict, self.movie_id_to_col_dict = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()

        self.predictor = ClusteringPredictor(self.ratings_matrix, self.row_similarity_matrix, self.col_similarity_matrix, self.new_dim, self.k_neighbors, self.randomized, self.randomized_num_extractions, self.random_seed)

        self.predictor.perform_precomputations()

    def predict(self):
        """
        Makes predictions based on clustering collaborative filtering.
        """
        PredictionStrategy.predict(self)
        return self.__predict(self.user_movie_instances_to_be_predicted)

    def __predict(self, instances_to_be_predicted: (int, int)) -> dict:
        """
        Predicts the ratings for the provided instances.
        The provided instances should be a list of (user_id, movie_id) tuples.
        The returned predictions are a dictionary, index by the (user_id, movie_id) tuples, containing the predicted ratings.

        :param instances_to_be_predicted: the list of (user_id, movie_id) tuples to make predictions from
        :return: the dictionary containing the predicted ratings, indexed by the user_id, movie_id tuples
        """

        return predict_instances_based_on_predictor(self.predictor, instances_to_be_predicted, self.user_id_to_row_dict,
                                                    self.movie_id_to_col_dict, transpose=False)
