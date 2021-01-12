from data_handling.DataLoader import DataLoader

# Output
from data_handling.DataPathProvider import DataPathProvider
from data_handling.LocalFileCsvProvider import LocalFileCsvProvider

import pandas as pd

WORKING_DIR = './data/evalutation'
TRAIN_DATA_FILENAME = 'ratings_train.csv'
TEST_PREDICTIONS_FILENAME = 'predictions_test.csv'
EXPECTED_PREDICTED_RATINGS = 'expected_predictions_test.csv'
REDUCED_MOVIES = 'reduced_movies.csv'
REDUCED_USERS = 'reduced_users.csv'

# Input
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = 'data/submissions/submission.csv'

# Create a data path provider
data_path_provider = DataPathProvider(movies_path=movies_file, users_path=users_file, ratings_path=ratings_file, predictions_path=predictions_file, submission_path=submission_file)

# Creata a data loader
data_loader = DataLoader(data_path_provider=data_path_provider, csv_provider=LocalFileCsvProvider())




def split_dataset_reduce(data_loader: DataLoader, users_filename: str, movies_filename: str, train_filename: str, test_filename: str, expected_ratings_filename: str, ratio: float = 0.3, frac_users: float = 1.0, frac_movies: float = 1.0):
    """
    Splits the data of the provided data loader into a training set and a test set. The method generates 3 csv files inside
    the evaluation folder. The first one contains the training data (user_id, movie_id, ratings), the second one contains the test data (i.e. the predictions
    the algorithm should make). The last file contains the expected ratings for the predictors.

    :param data_loader: - the data_loader containing the data to be split
    :param train_filename: - the name of training data file to be created
    :param test_filename: - the name of the test data file to be created
    :param expected_ratings_filename: - the name of the file containing the expected ratings
    :param ratio: - the percentage of the data to be used for testing and training
    """

    # shrink the size of the dataset

    user_data = data_loader.get_users_data()
    movie_data = data_loader.get_movies_data()
    ratings_data = data_loader.get_ratings_data()




    user_data_reduced = user_data.sample(frac=frac_users)
    movie_data_reduced = movie_data.sample(frac=frac_movies)


    # shrink the ratings data to only contain the ids of the selected users and movies
    user_id_reduced = user_data_reduced.iloc[:, [0]].to_numpy().T[0]
    movie_id_reduced = movie_data_reduced.iloc[:, [0]].to_numpy().T[0]

    print(movie_id_reduced)



    ratings_data_reduced = ratings_data[(ratings_data['userID'].isin(user_id_reduced)) & (ratings_data['movieID'].isin(movie_id_reduced))]

    print(ratings_data_reduced)


    # split ratings data into train and test data

    test_data = ratings_data_reduced.sample(frac=ratio)
    train_data = ratings_data_reduced.drop(test_data.index)
    predictions_to_be_performed = test_data.iloc[:, [0, 1]]

    # Write to file

    user_data_reduced.to_csv('{}/{}'.format(WORKING_DIR, users_filename), sep=';', index=False, header=False)
    movie_data_reduced.to_csv('{}/{}'.format(WORKING_DIR, movies_filename), sep=';', index=False, header=False)
    train_data.to_csv('{}/{}'.format(WORKING_DIR, train_filename), sep=';', index=False, header=False)
    test_data.to_csv('{}/{}'.format(WORKING_DIR, expected_ratings_filename), sep=';', index=False, header=False)
    predictions_to_be_performed.to_csv('{}/{}'.format(WORKING_DIR, test_filename), sep=';', index=False, header=False)


split_dataset_reduce(data_loader, REDUCED_USERS, REDUCED_MOVIES, TRAIN_DATA_FILENAME, TEST_PREDICTIONS_FILENAME, EXPECTED_PREDICTED_RATINGS, ratio=0.3, frac_users=0.1, frac_movies=0.1)
