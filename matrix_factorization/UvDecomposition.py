import numpy as np
import RMSE

class UvDecomposer():
    """
    Makes predictions by performing an UV decomposition on the provided matrix
    """

    def __init__(self, d:float, m_rating_matrix:np.array):
        # d cant be 1 or less, or greater than M dimensions
        assert(d > 1)
        assert(d < len(m_rating_matrix) and d < len(m_rating_matrix[0]) )


        self.zero_values = np.where(M == 0)  #find the undefined values

        self.M, self.avg_users, self.avg_items = self.__normalize(m_rating_matrix) #normalize given matrix m

        #Create U and V
        #   U is a n x d matrix
        #   V is a d x n matrix
        
        #Note: since M is normalized we create 0 matrices since sqrt(avg/d) is always 0
        # and for randomness move em around a beteen uniform
        # variable (-1,1) with this formula: (b - a) * random((num_row, num_col)) + a

        rng = np.random.default_rng()
        self.U = 2 * rng.random((len(self.M[0]), d)) -1
        self.V = 2 * rng.random((d, len(self.M)))    -1
        self.number_of_values = (len(self.M) * len(self.M[0])) - len(self.zero_values)

    def add_data_loader(self, data_loader: int):
        #self.ratings_matrix = data_loader.get_ratings_matrix()
        return 4

    def predict(self, iter:int):
        """
        Start creating prediciton matrix given 
        a number of iterations

        :param iter: number of iterations to run for
        """
        return self.__perform_decomposition(iter)


    def get_user_movie_rating(self, row:int, col:int) -> (float):
        """
        Get a single prediction (quite inefficient)

        :param row: row where value resides
        :param col: col where value resides

        :return: prediction
        """

        return np.matmul(self.U, self.V)[row][col] + ((self.avg_users[row] + self.avg_items[col])/2)

    def get_prediction_matrix(self) -> (np.array):
        """
        Get the full prediction matrix

        :return: prediction matrix
        """

        predictions = np.matmul(self.U, self.V)

        #now fill up prediction matrix with actual predictions
        for row in range(len(self.M)):
            for col in range(len(self.M[0])):
                predictions[row][col] = predictions[row][col] + ((self.avg_users[row] + self.avg_items[col])/2)

        return predictions 

    def RMSE(self) -> (float):
        """

        Creates an error vector with the Root-Mean-Square-Error
        measuring criteria.

        :return: RMSE value
        """

        sum_matrix = self.M - np.matmul(self.U, self.V)  #substract UV from M
        sum_matrix[self.zero_values] = 0                 #set the 0 values to 0 again
        
        #square the matrix, sum the rows, sum the vectors, divide by
        # number of non-zero-entries and take the square root
        return np.sqrt((np.square(sum_matrix)).sum(axis = 1).sum()/self.number_of_values)

    def __normalize(self, M:np.array) -> (np.array, np.array, np.array):
        """
        Normalizes given matrix M

        :return: Normalized matrix
        :return: Average of rows
        :return: Average of cols
        """
        
        #boolean matrix to get sum of non-zero values 
        # to calcualte the averages
        non_zeros = M > 0 
        users_i = non_zeros.sum(axis=1)
        items_j = non_zeros.sum(axis=0)


        #divide sum of user rating by number of times user gave a rating
        # divide sum of movie rating by number of times movie was rated 
        avg_users = M.sum(axis=1)/users_i
        avg_items = M.sum(axis=0)/items_j


        normalized_m = np.zeros((len(M), len(M[0])))
        #now fill up M matrix with normalized values
        for row in range(len(M)):
            for col in range(len(M[0])):
                #substact entry m_ij by (avg_user+avg_itme)/2
                normalized_m[row][col] = M[row][col] - (avg_users[row]+avg_items[col])/2
    
        normalized_m[self.zero_values] = 0

        return normalized_m, avg_users, avg_items

    def __perform_decomposition(self, iter:int):
        """
        Performs UV decomposition on the ratings matrix.
        Generate prediction matrix UV

        :param iter: numer of iterations to run 
        """

        #randomly choosing rows
        row_values = np.arange(5)
        np.random.shuffle(row_values)

        #randomly choosing columns
        col_values = np.arange(2)
        np.random.shuffle(col_values)

        #Gradient descent process for k iterations
        for k in range(iter):
            for i in row_values:
                for j in col_values:
                    self.__decompose_matrix_u(i, j)
                    self.__decompose_matrix_v(j, i)
            print("Iteration " + str(k+1) + ": RMSE Score => " ,self.RMSE())        




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
        denom = np.square(self.V[col]) #col == d
        denom[zeroes] = 0
        denom = denom.sum()
        
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

        denom = np.square(self.U[:,row])  ## row == d
        denom[zeroes] = 0
        denom = denom.sum()

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





M = np.array([[5,2,4,4,3], 
              [3,1,2,4,1], 
              [2,0,3,1,4], 
              [2,5,4,3,5],
              [4,4,5,4,0]])

M = M.astype(float)


decomposer = UvDecomposer(2,M)
decomposer.predict(8)
print()
print(decomposer.get_prediction_matrix())
print()
