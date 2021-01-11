import numpy as np

import sys

sys.path.append('.')

from FormulaFactory import FormulaFactory, ScoringMeasureType
from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
from matrix_factorization import MatrixNormalize

class UvDecomposer():
    """
    Makes predictions by performing an UV decomposition on the provided matrix
    """

    def __init__(self, d:int, iterations:int, delta:float, scorer_type: ScoringMeasureType,formula_factory: FormulaFactory):
        # d cant be 1 or less
        assert(d > 1)
        
        self.d = d                      #dimensions of the thin part
        self.delta = delta              #speed of descent
        self.iterations = iterations    #number of iterations to go for
        self.scoring_measure = formula_factory.create_scoring_measure(scorer_type)
        

    def add_data_loader(self, data_loader: DataLoader):

        PredictionStrategy.add_data_loader(self, data_loader)

        # vectors encoding the user ids and movie ids for each rows and columns
        #self.user_id_data, self.movie_id_data = self.data_loader.get_rating_matrix_user_and_movie_data()

        # dictionaries translating from user ids to rows and movie ids to columns
        self.user_id_to_row, self.movie_id_to_column = self.data_loader.get_rating_matrix_user_and_movie_index_translation_dict()


        # load the ratings matrix
        self.ratings_matrix = self.data_loader.get_ratings_matrix()

        #d can't be greater than the dimensions of M
        assert(d < len(self.ratings_matrix) and d < len(self.ratings_matrix[0]))

        #Find the zero values
        self.zero_values = np.where(self.ratings_matrix == 0)  #find the undefined values

        #Return normalized matrix M, and the avg_usr and avg_item vectors
        self.M, self.avg_users, self.avg_items = MatrixNormalize.normalize(self.ratings_matrix, self.zero_values)


        #Create U and V
        #   M us a n x m matrix
        #   U is a n x d matrix
        #   V is a d x m matrix
        
        #Note: since M is normalized we create 0 matrices since sqrt(avg/d) is always 0
        # and for randomness move them around with uniform
        # variable U(-1,1) with this formula: (b - a) * random((len(row), len(col))) + a

        rng = np.random.default_rng()
        self.U = 2 * rng.random((len(self.M[0]), d)) -1
        self.V = 2 * rng.random((d, len(self.M)))    -1

        self.number_of_values = len(self.M) * len(self.M[0]) - len(self.zero_values)


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

    def get_predictions(self, user_row:int, item_col:int) -> dict:
        """
        Get the prediction for the user and movie

        :return: prediction matrix
        """
        return self.predictions[user_row][item_col] + ((self.avg_users[user_row] + self.avg_items[item_col])/2)

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

        #randomly choosing rows
        row_values = np.arange(len(self.U))
        np.random.shuffle(row_values)

        #randomly choosing columns
        col_values = np.arange(self.d)
        np.random.shuffle(col_values)

        #Gradient descent process for k iterations
        for k in range(self.iterations):
            for i in row_values:
                for j in col_values:
                    self.__decompose_matrix_u(i, j)
                    self.__decompose_matrix_v(j, i)
            print("Iteration " + str(k+1) + ": Score => " ,self.score())        

        self.predictions = np.matmul(self.U, self.V) #generate the predictions


    def __decompose_matrix_u(self, row:int, col:int):
        """
        Perfoms the decompostion to optimize matrix U 
        on a given x value on the matrix

        :param row: row where x value is
        :param col: column where the x vallue is
        """

        m_rj = self.M[row]            #get the respective row
        zeroes = np.where(m_rj == 0)  #find the zeroes to ignore them

        #Denominator of the equation
        denom         = np.square(self.V[col]) #col == d
        denom[zeroes] = 0
        denom         = denom.sum()
        
        row_u = np.copy(self.U[row])    #get the row of U
        row_u[col] = 0                  #where the variable is located we set it to 0

        #Now we get the sum of multipling the selected u_row with the transpose
        sums = (row_u * self.V.transpose())
        
        #substract from its respective m_rj
        sums = sums.sum(axis=1)
        sums[zeroes[0]] = 0
        sums = m_rj - sums


        #Numerator of the equation#
        numer = (self.V[col] * sums).sum()   #multply row elements with sums, and sum it together

        self.U[row][col] = numer / denom

    def __decompose_matrix_v(self, row:int, col:int):
        """
        Perfoms the decompostion to optimize matrix V 
        on a given y value on the matrix

        :param row: row where y value is
        :param col: column where the y vallue is
        """

        m_is = self.M[:,col]           #get the respective col
        zeroes = np.where(m_is == 0)   #find the zeroes to ignore them

        denom         = np.square(self.U[:,row])  ## row == d
        denom[zeroes] = 0
        denom         = denom.sum()

        v_col = np.copy(self.V[:,col])     #get the respective row of v
        v_col[row] = 0

        #Now we get the sum of multipling the selected v_col with the U
        sums = (v_col * self.U) 

        #substract from its respective m_is
        sums = sums.sum(axis=1)
        sums[zeroes[0]] = 0
        sums = m_is - sums
        
        numer = (self.U[:,row]*sums).sum()
        
        self.V[row][col] = numer / denom

    def perform_precomputations(self):
        """
        Empty since catalin said to not precompute stuff
        since we are using the pickle library
        """
        pass

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

        self.__perform_decomposition()

        predictions_num = len(instances_to_be_predicted)
        num_prediction = 0

        for user_id, movie_id in instances_to_be_predicted:
            num_prediction += 1
            print('Progress {} / {}'.format(num_prediction, predictions_num))

            column = self.user_id_to_row[user_id]
            row    = self.movie_id_to_col[movie_id]
            

            rating = self.predictor.predict(row, column)

            predictions[(user_id, movie_id)] = rating

        print("Finished predictions!")

        return predictions


'''

M = np.array([[5,2,4,4,3], 
              [3,1,2,4,1], 
              [2,0,3,1,4], 
              [2,5,4,3,5],
              [4,4,5,4,0]])

M = M.astype(float)

# Where data is located
movies_file      = '../data/movies.csv'
users_file       = '../data/users.csv'
ratings_file     = '../data/ratings.csv'
predictions_file = '../data/predictions.csv'
submission_file  = '../data/submission.csv'


from data_handling.DataPathProvider import DataPathProvider
from data_handling.LocalFileCsvProvider import LocalFileCsvProvider

# Create a data path provider
data_path_provider = DataPathProvider(movies_path=movies_file, users_path=users_file, ratings_path=ratings_file, predictions_path=predictions_file, submission_path=submission_file)

# Creata a data loader
data_loader = DataLoader(data_path_provider=data_path_provider, csv_provider=LocalFileCsvProvider())




decomposer = UvDecomposer(2,M)
decomposer.add_data_loader(data_loader)

print(decomposer.user_id_data)
'''