from commons.FormulaFactory import SimilarityMeasureType, FormulaFactory
from commons.PredictionStrategy import PredictionStrategy
from collaborative_filtering.Utils import predict_instances_based_on_predictor
from collaborative_filtering.lsh.CosineDistanceLsh import CosineDistanceLsh
from collaborative_filtering.lsh.LocalitySensitiveHashTable import LocalitySensitiveHashTable
from collaborative_filtering.lsh.LshCollaborativeFilteringPredictor import LshCollaborativeFilteringPredictor


class UserLshCollaborativeFiltering(PredictionStrategy):
    def __init__(self, k_neighbors: int, signiture_length: int, max_query_distance: int,
                 formula_factory: FormulaFactory, random_seed: int, cosine_similarity_type: SimilarityMeasureType = SimilarityMeasureType.MEANLESS_COSINE_SIMILARITY):

        self.signiture_length = signiture_length
        self.k_neighbors = k_neighbors
        self.max_query_distance = max_query_distance
        self.formula_factory = formula_factory
        self.random_seed = random_seed
        self.cosine_similarity_type = cosine_similarity_type

    def perform_precomputations(self):
        PredictionStrategy.perform_precomputations(self)

        self.ratings_matrix = self.data_loader.get_ratings_matrix()

        self.user_id_vector, self.movie_id_vector = self.data_loader.get_rating_matrix_user_and_movie_data()
        self.user_id_to_row_dict, self.movie_id_to_col_dict = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()

        self.lsh_table = None

        if self.disk_persistor is None:

            self.lsh_table: LocalitySensitiveHashTable = \
                CosineDistanceLsh(self.ratings_matrix, self.signiture_length, self.random_seed)

        else:
            results = self.disk_persistor.perist_computation(
                computations=[(lambda: CosineDistanceLsh(self.ratings_matrix, self.signiture_length, self.random_seed),
                               self.persistence_id)],
                force_update=self.force_update
            )

            self.lsh_table: LocalitySensitiveHashTable = results[0]


        # init the lsh predictor
        self.predictor = LshCollaborativeFilteringPredictor(self.ratings_matrix, self.k_neighbors, self.max_query_distance,
                                                            self.cosine_similarity_type,
                                                            self.lsh_table, self.formula_factory)


    def predict(self):
        """
        Makes predictions based on user-user collaborative filtering.
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
