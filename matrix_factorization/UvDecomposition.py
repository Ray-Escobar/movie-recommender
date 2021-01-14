import numpy as np
import math
import sys

sys.path.append('.')

from FormulaFactory import FormulaFactory, ScoringMeasureType
from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
from matrix_factorization import MatrixNormalize

class UvDecomposer(PredictionStrategy):
    """
    Makes predictions by performing an UV decomposition on the provided matrix
    """

    def __init__(self, d:int, iterations:int, mu:float, formula_factory:FormulaFactory,scorer_type: ScoringMeasureType):

        # d cant be 1 or less
        assert(d > 1)
        
        self.d = d                      #dimensions of the thin part
        self.mu = mu              #speed of descent
        self.iterations = iterations    #number of iterations to go for
        self.scoring_measure = formula_factory.create_scoring_measure(scoring_measure_type=scorer_type)

    def get_prediction_matrix(self) -> (np.array):
        """
        Get the full prediction matrix

        :return: prediction matrix
        """

        predictions = np.matmul(self.U, self.V)

        #now fill up prediction matrix with actual predictions by removing normalization
        for row in range(len(self.M)):
            for col in range(len(self.M[0])):
                predictions[row][col] = predictions[row][col] + ((self.avg_users[row] + self.avg_items[col])/2)

        return predictions

    def __predict_score(self, user_row:int, item_col:int) -> dict:
        """
        Get the prediction for the user and movie

        :return: prediction matrix
        """
        return self.predictions[user_row][item_col] #+ ((self.avg_users[user_row] + self.avg_items[item_col])/2)

    def score(self) -> (float):
        """
        Calcualtes error of the UV matrix

        :return: score value from scoring measure
        """
        return self.scoring_measure(self.M, np.matmul(self.U, self.V), self.zero_values, self.number_of_values)

    def __perform_decomposition(self):
        """
        Performs UV decomposition on the ratings matrix.
        Generate prediction matrix UV

        :param iterations: numer of iterations to run 
        """

        indices = np.nonzero(self.M)

        #Gradient descent process for k iterations
        for k in range(self.iterations):
            for row, col in zip(indices[0], indices[1]):
                self.decompose_matrices(row, col)
            print("Iteration " + str(k+1) + ": Score => " ,self.score())        

        self.predictions = np.matmul(self.U, self.V) #generate the predictions



    def decompose_matrices(self, row:int, col:int):
        """
        Methods have to implement this function
        """
        raise Exception('You must define a decompose matrices function')

        #gradient = 2*(self.M[row][col] - np.dot(self.U[row], self.V[:,col]))
        #self.U[row]   = self.U[row]   + self.mu*(gradient*self.V[:,col] - self.regul * self.U[row]  )
        #self.V[:,col] = self.V[:,col] + self.mu*(gradient*self.U[row]   - self.regul * self.V[:,col])


    def perform_precomputations(self):
        """
        Perform precomputations
        """

        PredictionStrategy.perform_precomputations(self)

        # dictionaries translating from user ids to rows and movie ids to columns
        self.user_id_to_row, self.movie_id_to_col = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()

        # load the ratings matrix
        self.M = self.data_loader.get_ratings_matrix()
        
        #d can't be greater than the dimensions of M
        assert(self.d < len(self.M) and self.d < len(self.M[0]))

        self.zero_values = np.where(self.M == 0) 

        #Create U and V
        #   M us a n x m matrix (the rating matrix)
        #   U is a n x d matrix
        #   V is a d x m matrix
        
        self.U = np.random.rand(len(self.M), self.d)
        self.V = np.random.rand(self.d, len(self.M[0]))

        self.number_of_values = len(self.M[np.where(self.M > 0)])

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

        print("Starting predictions with UV decomposition...")
        self.__perform_decomposition() #start UV decompositon

        predictions_num = len(instances_to_be_predicted)
        num_prediction = 0

        for user_id, movie_id in instances_to_be_predicted:
            num_prediction += 1
            print('Progress {} / {}'.format(num_prediction, predictions_num))

            row = self.user_id_to_row[user_id]
            col = self.movie_id_to_col[movie_id]
            

            rating = self.__predict_score(row, col)
            print(rating)

            predictions[(user_id, movie_id)] = rating

        print("Finished predictions!")
        
        return predictions