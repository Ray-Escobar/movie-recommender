'''
Matrix M = 
| 5  2  4  4  3
| 3  1  2  4  1 
| 2     3  1  4
| 2  5  4  3  5
| 4  4  5  4  


- we substract M by our UV matrix
- we square and sum these to get the RMSE
'''

import numpy as np

def RMSE(UV:np.array, M:np.array, n:int, zero_values:np.array) -> (np.array):
        """

        Creates an error vector with the Root-Mean-Square-Error
        measuring criteria.

        :param UV: our approximation of factorization M
        :param M : our matrix M with user ratings for each movie
        :param n : number of non-zero values in the matrix
        :param zero_values : where the zero values are located in M
        :return: a vector containing the RMSE for each row
        """

        sum_matrix = M-UV           #substract UV from M
        sum_matrix[zero_values] = 0 #set the 0 values to 0 again
        
        #square the matrix, sum the rows, sum the vectors, divide by
        # number of non-zero-entries and take the square root
        return np.sqrt((np.square(sum_matrix)).sum(axis = 1).sum()/n)

        

