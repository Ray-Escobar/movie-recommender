import numpy as np
import RMSE

class UvDecomposer():
    """
    Makes predictions by performing an SVD decomposition on the provided matrix
    """

    def __init__(self, u_matrix:np.array, v_matrix:np.array, m_rating_matrix:np.array):
        self.M = m_rating_matrix
        self.U = u_matrix
        self.V = v_matrix
        #self.uv_matrix = np.matmul(U,V)
        self.zero_values = np.where(M == 0 )
        #self.number_of_values = np.count_nonzero(M)
        self.number_of_values = (len(M) * len(M[0])) - len(self.zero_values)

    def add_data_loader(self, data_loader: int):
        #self.ratings_matrix = data_loader.get_ratings_matrix()
        return 4

    def predict(self):
        return self.__perform__decomposition()

    def RMSE(self) -> (np.array):
        """

        Creates an error vector with the Root-Mean-Square-Error
        measuring criteria.

        :return: RMSE value
        """

        sum_matrix = self.M - np.matmul(self.U, self.V)  #substract UV from M
        sum_matrix[self.zero_values] = 0                     #set the 0 values to 0 again
        
        #square the matrix, sum the rows, sum the vectors, divide by
        # number of non-zero-entries and take the square root
        return np.sqrt((np.square(sum_matrix)).sum(axis = 1).sum()/self.number_of_values)

    def perform_uv_decomposition(self, iter:int):
        """
        Performs UV decomposition on the ratings matrix.
        Generate prediction matrix UV

        :param iter: numer of iterations to run 
        """
        for i in range(5):
            for j in range (2):
                #print("i: ", i)
                #print("j: ", j)
                self.decompose_matrix_U(i, j)
                self.decompose_matrix_V(j, i)

        print(self.U)
        print(self.V)
        print()
        print(np.matmul(self.U,self.V))



    def decompose_matrix_U(self, row:int, col:int):

        m_rj = self.M[row] #get the respective row
        zeroes = np.where(m_rj == 0)     #find the zeroes to ignore them

        #Denominator of the equation#
        denom = np.square(self.V[col]) #col == d
        denom[zeroes] = 0
        denom = denom.sum()
        
        row_u = np.copy(self.U[row]) #get the row of U
        row_u[col] = 0                      #where the variable is we set it to 0

        #Now we get the sum of multipling the selected u_row with the transpose
        sums = (row_u * self.V.transpose())
        
        #substract from its respective m_rj
        sums = sums.sum(axis=1)
        sums[zeroes[0]] = 0
        sums = m_rj - sums


        #Numerator of the equation#
        numer = (self.V[col] * sums).sum()   #multply row elements with sums, and sum it together


        self.U[row][col] = numer / denom
        #print(self.U)

    def decompose_matrix_V(self, row:int, col:int):


        m_is = self.M[:,col] #get the respective col
        zeroes = np.where(m_is == 0)       #find the zeroes to ignore them

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

        #print(self.V)



'''
Matrix M = 
| 5  2  4  4  3
| 3  1  2  4  1 
| 2  0  3  1  4
| 2  5  4  3  5
| 4  4  5  4  0
'''


M = np.array([[5,2,4,4,3], 
              [3,1,2,4,1], 
              [2,0,3,1,4], 
              [2,5,4,3,5],
              [4,4,5,4,0]])

U = np.array([[1,1], 
              [1,1],
              [1,1], 
              [1,1], 
              [1,1]])

V = np.array([[1, 1, 1, 1, 1], 
              [1, 1, 1, 1, 1]])
 

M = M.astype(float)
U = U.astype(float)
V = V.astype(float)

'''
U = np.array([[2,5]])

uuu = np.array([[2,3], 
                [3,1],
                [4,6], 
                [3,2], 
                [5,7]])

vvv = np.array([[2, 3, 4, 3, 5], 
                [3, 1, 6, 2, 7]])

print(vvv.transpose() * u)
'''


decomposer = UvDecomposer(U,V,M)

print("Initial RMSE: ", decomposer.RMSE())
decomposer.perform_uv_decomposition(100)
print("Posterior RMSE: ",decomposer.RMSE())

#decomposer.decompose_matrix_U(0,0)
#decomposer.decompose_matrix_V(0,0)
#decomposer.decompose_matrix_U(2,0) #row 2 column 1
#decomposer.decompose_matrix_U(0,1)

#1.805787796286538
#1.805787796286538