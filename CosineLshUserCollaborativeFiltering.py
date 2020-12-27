from typing import List, Set, Tuple

from DataLoader import DataLoader
from FormulaFactory import FormulaFactory
from PredictionStrategy import PredictionStrategy
import numpy as np


class CosineLshUserCollaborativeFiltering(PredictionStrategy):
    """
    Makes predictions based on a more efficient collaborative filtering that uses and lsh to query for
    the closest users in terms of cosine distances.
    """

    def __init__(self, k_neighbors: int, signiture_length: int, max_query_distance: int,
                 formula_factory: FormulaFactory, random_seed: int):
        """
        Initializes the strategy with the provided parameters.
        :param k_neighbors: the number of neighbors to be used in collaborative filtering
        """

        np.random.seed(random_seed)

        self.k_neighbors = k_neighbors
        self.formula_factory = formula_factory
        self.signiture_length = signiture_length
        self.max_query_distance = max_query_distance

        self.meanless_cosine_sim = self.formula_factory.create_meanless_cosine_similarity_measure()
        self.sim_weight_avg = self.formula_factory.create_rating_average_weighted_by_similarity_function()





    def perform_precomputations(self):
        PredictionStrategy.perform_precomputations(self)

        self.ratings_matrix = self.data_loader.get_ratings_matrix()
        self.user_id_vector, self.movie_id_vector = self.data_loader.get_rating_matrix_user_and_movie_data()
        self.user_id_to_row_dict, self.movie_id_to_col_dict = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()

        self.lsh_table = None

        if self.disk_persistor is None:

            self.lsh_table = LSH(self.ratings_matrix,
                                self.user_id_vector, self.movie_id_vector,
                                self.user_id_to_row_dict, self.movie_id_to_col_dict,
                                self.signiture_length)

        else:
            results = self.disk_persistor.perist_computation(
                computations=[(lambda: LSH(self.ratings_matrix,
                                self.user_id_vector, self.movie_id_vector,
                                self.user_id_to_row_dict, self.movie_id_to_col_dict,
                                self.signiture_length), self.persistence_id)],
                force_update=self.force_update
            )

            self.lsh_table = results[0]



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

        predictions = dict()

        print("Starting predictions...")

        predictions_num = len(instances_to_be_predicted)
        num_prediction = 0

        for user_id, movie_id in instances_to_be_predicted:

            num_prediction += 1
            print('Progress {} / {}'.format(num_prediction, predictions_num))

            rating = self.__predict_rating_for(user_id, movie_id)

            predictions[(user_id, movie_id)] = rating

        print("Finished predictions!")

        return predictions

    def __predict_rating_for(self, target_user_id, target_movie_id):
        """
        Makes a prediction about the rating of the target user for the target movie.

        :param target_user_id:
        :param target_movie_id:
        :return: the predicted rating of the target user in regard to the target movie
        """

        target_user_index = self.user_id_to_row_dict[target_user_id]
        target_user = self.ratings_matrix[target_user_index, :]

        # get the neighbors of the user
        neighbour_user_ids = self.lsh_table.query_neighbors(target_user, self.k_neighbors, target_movie_id,
                                                            self.max_query_distance)


        # if no neighbors can be found, avoid making a prediction (return 0)
        if len(neighbour_user_ids) == 0:
            return 0.0

        # get the sim-user tuples list
        sim_user_tuple_list = self.__get_similarity_rating_tuples_from_ids(target_movie_id, target_user, neighbour_user_ids)

        # compute the prediction for the rating as a weighted average of the neighbor user ratings

        rating = self.sim_weight_avg(sim_user_tuple_list)

        if np.abs(rating) < 0.01:
            return 0


        # deal with ratings outside the range
        if rating < 1:
            rating = 1.0
        elif rating > 5:
            rating = 5.0
        return rating

    def __get_similarity_rating_tuples_from_ids(self, target_movie_id, target_user: np.array, neighbour_user_ids: List[int]) -> List[
        Tuple[float, float]]:
        """
        Computes the similarities between the target_user and the users corresponding the the provided user ids.

        :param target_user: the user for which we compute similarities to users in the matrix
        :param user_ids: the list of user ids to be compared with our target user.
        :return: a list of tuples containing similarity-user_data tuples
        """
        similarity_user_tuples: List[Tuple[float, np.array]] = list()


        for user_id in neighbour_user_ids:
            user_row = self.user_id_to_row_dict[user_id]
            movie_col = self.movie_id_to_col_dict[target_movie_id]
            neighbor_user = self.ratings_matrix[user_row]

            # compute the meanless cosine similarity between target_user and the neighbor_user
            similarity_value = self.meanless_cosine_sim(target_user, neighbor_user)


            similarity_user_tuples.append((similarity_value, neighbor_user[movie_col]))

        return similarity_user_tuples


class LSH:
    """
    LSH data structure that can be used to query the k nearest neighbors
    """

    def __init__(self, ratings_matrix: np.array, user_id_vector: np.array, movie_id_vector: np.array,
                 user_id_to_row_dict: dict,
                 movie_id_to_column_dict: np.array,
                 signiture_length: int):
        """
        Initializes a new LSH of the provided ratings_matrix.

        :param ratings_matrix:  the ratings matrix create the LSH for
        :param user_id_vector: a vector specifying the user id corresponding to each row
        :param movie_id_vector: a vector specifying the movie id for each column
        :param user_id_to_row_dict: dictionary translating user ids to rows in the rating matrix
        :param movie_id_to_column_dict: dictionary translating the movie ids to columns in the ratings matrix
        :param signiture_length: specifies the length of the signitures to be computed
        :param formula_factory: factory for creating useful formulas
        """
        self.ratings_matrix: np.array = ratings_matrix
        self.user_id_vector: np.array = user_id_vector
        self.movie_id_vector: np.array = movie_id_vector
        self.user_id_to_row_dict = user_id_to_row_dict
        self.movie_id_to_column_dict = movie_id_to_column_dict
        self.signiture_length: int = signiture_length

        # generate the planes used in the locality sensitive hashing
        num_users: int = self.ratings_matrix.shape[0]
        num_movies: int = self.ratings_matrix.shape[1]
        self.planes = self.__generate_k_random_planes(k=self.signiture_length, dim=num_movies)

        self.lsh_map: dict = self.__generate_locality_sensitive_hash_table(self.planes)

    def query_neighbors(self, user: np.array, k_neighbors: int, target_movie_id: int, max_distance: int) -> List[int]:
        """
        Returns the ids of the k most similar users to the given user that have rated the target movie.

        :param user: the user to find similar users for
        :param k_neighbors: the number of neighbor users to be returned
        :param target_movie_col: the column of the movie that should be rated by all neighbor users
        :param max_distance: the maximum search distance
        :return: a list containing a maximum of k neighbors similar to the target user, that are within the max_distance from it
        and have rated the target movie
        """
        neighbors: List[int] = []

        target_user_signiture = self.__generate_signiture(user, self.planes)

        distance = 0

        while len(neighbors) < k_neighbors and distance < max_distance:
            # get the neighbors at the currently selected distance in the hash_map
            nearby_neighbors: Set[np.array] = set()

            if target_user_signiture - distance in self.lsh_map.keys():
                nearby_neighbors.update(self.lsh_map[target_user_signiture - distance])

            if target_user_signiture + distance in self.lsh_map.keys():
                nearby_neighbors.update(self.lsh_map[target_user_signiture + distance])

            # add only the neighbors that contain ratings for the target movie
            for nearby_neighbor in nearby_neighbors:
                if self.__user_rated_target_movie(nearby_neighbor, target_movie_id):
                    neighbors.append(nearby_neighbor)

                # if the maximum number of neighbors has been reached, stop adding neighbors
                if len(neighbors) >= k_neighbors:
                    break

            # increase the search distance by 1
            distance += 1

        return neighbors

    def __user_rated_target_movie(self, user_id: int, target_movie_id: int) -> bool:
        """
        Returns true if the user has rated the target movie, false otherwise.

        :param user_id: the id of the user to be checked
        :param target_movie_id: the id of the movie to be checked

        :return: true, if the provided user has rated the provided movie, false otherwise
        """

        user_row = self.user_id_to_row_dict[user_id]
        user = self.ratings_matrix[user_row, :]

        movie_col = self.movie_id_to_column_dict[target_movie_id]



        if np.abs(user[movie_col]) < 0.01:
            return False

        return True

    def __generate_locality_sensitive_hash_table(self, planes: List[np.array]) -> dict:
        """
        Generates a multi-map (dictionary) containing all user ids indexed by their lsh signiture.

        :return: a dictionary as described above
        """

        print("Generating cosine lsh signatures...")

        num_users: int = self.ratings_matrix.shape[0]

        bins: dict = dict()

        for user_row in range(num_users):
            print('Progress ' + str(user_row + 1) + " / " + str(num_users))

            user: np.array = self.ratings_matrix[user_row, :]

            user_signiture: int = self.__generate_signiture(user, planes)

            if user_signiture not in bins.keys():
                bins[user_signiture] = list()

            bins[user_signiture].append(self.user_id_vector[user_row])

        print("Finished generating signatures.")

        return bins

    def __generate_k_random_planes(self, k: int, dim: int) -> List[np.array]:
        """
        Generates a list of k dim -dimensional random planes samples from a normal distribution
        :param k: the number of planes to be generated
        :param dim: the dimension of the planes to be generated
        :return: a list of k randomly generated planes
        """

        planes = []


        for _ in range(k):
            plane = np.random.randn(dim)
            planes.append(plane)


        return planes

    def __generate_signiture(self, user: np.array, planes: List[np.array]) -> int:
        """
        Computes the LSH signiture of a user using the following method:
        https://bogotobogo.com/Algorithms/Locality_Sensitive_Hashing_LSH_using_Cosine_Distance_Similarity.php

        :param user: the user ratings vector to compute the signiture for
        :param planes: a list of randomly generated planes used to compute the signitures
        :return: a signiture-length bit number representing the signiture
        """

        signiture = 0


        for plane in planes:
            signiture = signiture << 1

            if np.dot(user, plane) > 0:
                signiture |= 1


        return signiture
