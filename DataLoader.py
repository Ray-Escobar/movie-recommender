from typing import List, Tuple

from CsvProvider import CsvProvider
from DataPathProvider import DataPathProvider
import pandas as pd
import numpy as np


class DataLoader:
    """
    Class responsible from loading the data into a pandas data frames.
    """
    def __init__(self, data_path_provider: DataPathProvider, csv_provider: CsvProvider):
        """
        Initializes a data loader object.
        :param data_path_provider: object used to determine the path to the csv data files
        """
        self.csv_provider = csv_provider
        self.data_path_provider = data_path_provider
        self.movies_data = None
        self.predictions_data = None
        self.ratings_data = None

        self.ratings_matrix = None
        self.user_id_to_ratings_matrix_row = None
        self.movie_id_to_ratings_matrix_column = None

        self.ratings_matrix_user_column = None
        self.ratings_matrix_movie_column = None
        self.ratings_map= None
        self.users_data = None

    def __load_data(self, path: str, column_names: List[str]) -> pd.DataFrame:
        return self.csv_provider.read_csv(path, delimiter=';', column_names = column_names)

    def __get_data(self, data: pd.DataFrame, path: str, column_names: List[str]) -> pd.DataFrame:
        if data is None:
            data = self.__load_data(path, column_names=column_names)

        return data.copy()

    def get_movies_data(self) -> pd.DataFrame:
        return self.__get_data(self.movies_data, self.data_path_provider.get_movies_path(), ['movieID', 'year', 'movie'])

    def get_predictions_data(self) -> pd.DataFrame:
        return self.__get_data(self.predictions_data, self.data_path_provider.get_predictions_path(), ['userID', 'movieID'])

    def get_prediction_instances(self)-> List[Tuple[int, int]]:
        prediction_data = self.get_predictions_data()
        return list(zip(prediction_data['userID'].tolist(), prediction_data['movieID'].tolist()))

    def get_ratings_data(self) -> pd.DataFrame:
        return self.__get_data(self.ratings_data, self.data_path_provider.get_ratings_path(), ['userID', 'movieID', 'rating'])

    def get_rating_matrix_user_and_movie_data(self) -> (List[int], List[int]):
        if self.ratings_matrix_user_column is not None and self.ratings_matrix_movie_column is not None:
            return self.ratings_matrix_user_column, self.ratings_matrix_movie_column

        users_data: pd.DataFrame = self.get_users_data()
        movie_data: pd.DataFrame = self.get_movies_data()


        user_column: List[int] = users_data['userID'].tolist()
        movie_column: List[int] = movie_data['movieID'].tolist()

        # get a sorted list of all unique user ids
        self.ratings_matrix_user_column = sorted(list(set(user_column)))

        # get a sorted list of all unique movie ids
        self.ratings_matrix_movie_column = sorted(list(set(movie_column)))

        return self.ratings_matrix_user_column, self.ratings_matrix_movie_column

    def get_rating_matrix_user_and_movie_index_translation_dict(self) -> (dict, dict):
        """
        Returns a tuple of dictionaries that can be used to translate user ids to rows inside the rating matrix
        and movie ids to columns inside the rating matrix.

        :return: a tuple of the 2 described dictionaries, the first for users, the second for movies
        """
        if self.user_id_to_ratings_matrix_row is not None and self.movie_id_to_ratings_matrix_column is not None:
            return self.user_id_to_ratings_matrix_row, self.movie_id_to_ratings_matrix_column

        user_data, movie_data = self.get_rating_matrix_user_and_movie_data()

        self.user_id_to_ratings_matrix_row = dict()
        self.movie_id_to_ratings_matrix_column = dict()

        for row, user_id in enumerate(user_data):
            self.user_id_to_ratings_matrix_row[user_id] = row

        for col, movie_id in enumerate(movie_data):
            self.movie_id_to_ratings_matrix_column[movie_id] = col

        return self.user_id_to_ratings_matrix_row, self.movie_id_to_ratings_matrix_column



    def get_ratings_map(self) -> dict:
        """
        Returns a map of the ratings, indexed by the (user_id, movie_id) pair
        :return: a tuple consisting of the ratings map, the user ids list and the movie ids list
        """


        if self.ratings_map is not None:
            return self.ratings_map


        ratings_data: pd.DataFrame = self.get_ratings_data()

        ratings_column: List[int] = ratings_data['rating'].tolist()

        user_column: List[int] = ratings_data['userID'].tolist()
        movie_column: List[int] = ratings_data['movieID'].tolist()


        # create a hash table, indexed by the user_id and movie_id, containing the ratings_column data
        ratings_map = dict()
        for user_id, movie_id, rating in zip(user_column, movie_column, ratings_column):
            ratings_map[(user_id, movie_id)] = rating


        self.ratings_map = ratings_map


        return self.ratings_map




    def get_ratings_matrix(self) -> (np.array, dict, dict):
        """
        Returns a matrix, with rows being represented by users, and the columns by the movies. The entries of the matrix will be the ratings.

        :return: a tuple consisting of the ratings matrix
        """

        # create the actual ratings matrix

        if self.ratings_matrix is not None:
            return self.ratings_matrix

        hash_table = self.get_ratings_map()
        user_column_unique, movie_column_unique = self.get_rating_matrix_user_and_movie_data()


        ratings_matrix:np.array= np.array(len(user_column_unique) * [len(movie_column_unique) * [0.0]]) # initialize a matrix of the right size, with 0 (symbol for unrated movies)

        for row, user_id in enumerate(user_column_unique):
            for column, movie_id in enumerate(movie_column_unique):
                if (user_id, movie_id) in hash_table.keys():
                    ratings_matrix[row][column] = hash_table[(user_id, movie_id)]



        self.ratings_matrix = ratings_matrix

        return self.ratings_matrix


    def get_users_data(self) -> pd.DataFrame:
        return self.__get_data(self.users_data, self.data_path_provider.get_users_path(), ['userID', 'gender', 'age', 'profession'])




