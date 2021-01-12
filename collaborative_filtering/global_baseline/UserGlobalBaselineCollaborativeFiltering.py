from typing import Tuple, List

from PredictionStrategy import PredictionStrategy
from collaborative_filtering.global_baseline.GlobalBaselineCollaborativeFilteringPredictor import \
    GlobalBaselineCollaborativeFilteringPredictor
from collaborative_filtering.naive.NaiveCollaborativeFilteringPredictor import NaiveCollaborativeFilteringPredictor
from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.Utils import predict_instances_based_on_predictor


class UserGlobalBaselineCollaborativeFiltering(PredictionStrategy):

    def __init__(self, k_neighbors: int, sim_matrix: RowPearsonSimilarityMatrix):
        self.k_neighbors = k_neighbors
        self.sim_matrix = sim_matrix

    def perform_precomputations(self):
        PredictionStrategy.perform_precomputations(self)

        self.ratings_matrix = self.data_loader.get_ratings_matrix()

        self.user_id_vector, self.movie_id_vector = self.data_loader.get_rating_matrix_user_and_movie_data()
        self.user_id_to_row_dict, self.movie_id_to_col_dict = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()

        self.predictor = GlobalBaselineCollaborativeFilteringPredictor(data_matrix=self.ratings_matrix,
                                                              row_similarity_matrix=self.sim_matrix,
                                                              k_neighbors=self.k_neighbors)

    def predict(self):
        """
        Makes predictions based on user-user collaborative filtering.
        """
        PredictionStrategy.predict(self)
        return self.__predict(self.user_movie_instances_to_be_predicted)

    def __predict(self, instances_to_be_predicted: List[Tuple[int, int]]) -> dict:
        """
        Predicts the ratings for the provided instances.
        The provided instances should be a list of (user_id, movie_id) tuples.
        The returned predictions are a dictionary, index by the (user_id, movie_id) tuples, containing the predicted ratings.

        :param instances_to_be_predicted: the list of (user_id, movie_id) tuples to make predictions from
        :return: the dictionary containing the predicted ratings, indexed by the user_id, movie_id tuples
        """

        return predict_instances_based_on_predictor(self.predictor, instances_to_be_predicted, self.user_id_to_row_dict,
                                                    self.movie_id_to_col_dict, transpose=False)
