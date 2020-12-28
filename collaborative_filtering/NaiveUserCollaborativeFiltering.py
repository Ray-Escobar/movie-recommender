from typing import List

from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
from FormulaFactory import FormulaFactory, SimilarityMeasureType
import numpy as np
import heapq

"""
Deprectated! Better prediction methods exist!/
"""
class NaiveUserCollaborativeFiltering(PredictionStrategy):
    def __init__(self, k_neighbors: int, similarity_measure_type: SimilarityMeasureType, formula_factory: FormulaFactory):
        self.k_neighbors: int = k_neighbors
        self.formula_factory = formula_factory
        self.similarity_measure = self.formula_factory.create_similarity_measure(similarity_measure_type)


    def add_data_loader(self, data_loader: DataLoader):
        PredictionStrategy.add_data_loader(self, data_loader)

        # vectors encoding the user ids and movie ids for each rows and columns
        self.user_id_data, self.movie_id_data = self.data_loader.get_rating_matrix_user_and_movie_data()

        # dictionaries translating from user ids to rows and movie ids to columns
        self.user_id_to_row, self.movie_id_to_column = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()


        # load the ratings matrix
        self.ratings_matrix = self.data_loader.get_ratings_matrix()

        # compute the similarity matrix
        self.similarity_matrix = self.__compute_similarity_matrix()




    def predict(self) -> dict:
        """
        Predicts the ratings for the instances provided by the data loader
        he returned predictions are a dictionary, index by the (user_id, movie_id) tuples, containing the predicted ratings.

        :return: the dictionary containing the predicted ratings, indexed by the user_id, movie_id tuples
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

        predictions = dict()

        for user_id, movie_id in instances_to_be_predicted:
            user_index = self.user_id_to_row[user_id]
            movie_index = self.movie_id_to_column[movie_id]

            rating = self.__predict_rating_for(user_index, movie_index)

            predictions[(user_id, movie_id)] = rating

        return predictions


    def __compute_similarity_matrix(self):
        """
        Computes the similarity matrix between the users.
        similarity_matrix[i][j] = the similarity between user i and user j

        :return: the computed similarity matrix
        """
        num_users = self.ratings_matrix.shape[0]

        # the simialrity matrix initializes as a num_users x num_users matrix filled with zeros
        similarity_matrix = np.array(num_users * [num_users * [0.0]])

        # compute the similarity between all possible user combinations
        for user1_index in range(num_users):
            for user2_index in range(num_users):
                user1_ratings = self.ratings_matrix[user1_index, :]
                user2_ratings = self.ratings_matrix[user2_index, :]

                similarity = self.similarity_measure(user1_ratings, user2_ratings)


                similarity_matrix[user1_index, user2_index] = similarity

        return similarity_matrix


    def __predict_rating_for(self, target_user_index, target_movie_index):
        """
        Makes a prediction about the rating of the target user for the target movie.

        :param target_user_index:
        :param target_movie_index:
        :return:
        """
        most_similar_neighbors = self.__find_k_most_similar_neighbors_with_rated_movie(target_user_index, target_movie_index)

        # if no most similar neighbors were found, return 0 (i.e. the rating cannot be predicted)
        if len(most_similar_neighbors) == 0:
            return 0.0

        # get the list of similarity-rating tuples for the retrieved neighbors
        sim_rating_tuples = []

        for sim, user_index in most_similar_neighbors:
            sim_rating_tuples.append((sim, self.ratings_matrix[user_index, target_movie_index]))

        print(sim_rating_tuples)

        # return the similarity weighted average of the ratings
        sim_avg = self.formula_factory.create_rating_average_weighted_by_similarity_function()
        return sim_avg(sim_rating_tuples)


    def __find_k_most_similar_neighbors_with_rated_movie(self, target_user_index, target_movie_index) -> List[int]:
        """
        Returns a list of similarity - index tuples for the most similar k users to the provided user, that have rated the provided movie.

        :param target_user_index: the row index of the user to find the neighbors for
        :param target_movie_index: the column index of the movie that should be rated by the neighbors
        :return: a list containing the similarity-index tuples of the users with the properties specified above. Ideally, the list will contain
                k elements, however, if less than k neighbors exist, then the list will be shorter
        """

        num_users = self.ratings_matrix.shape[0]

        most_similar_k_users = []

        for user_index in range(num_users):
            # we need to skip our target user
            if user_index == target_user_index:
                continue


            # we need to skip the users that don't have a rating for the target
            if self.__user_has_rating_for_movie(user_index, target_movie_index) is False:
                continue


            sim = self.similarity_matrix[target_user_index][user_index]

            # add the potential neighbor to the queue
            heapq.heappush(most_similar_k_users, (sim, user_index))

            # if there are more the number of neighbors has become larger than k, remove the most dissimilar neihbor
            if len(most_similar_k_users) > self.k_neighbors:
                heapq.heappop(most_similar_k_users)


        return most_similar_k_users



    def __user_has_rating_for_movie(self, user_index: int, movie_index: int) -> bool:
        """
        Returns true if the provided user has rated the provided movie.

        :param user_index: the index of the user to analyze
        :param movie_index: the index of the movie to check whether or not was rated by the user.

        :return: true, if the movie has a rating, false otherwise
        """
        if np.abs(self.ratings_matrix[user_index][movie_index]) < 0.01:
            return False

        return True


