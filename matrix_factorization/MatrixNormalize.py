import numpy as np

def normalize(M:np.array, zero_values:np.array) -> (np.array):
        """

        Normalizes a given matrix and returns a new normalized matrix with
        the average of rows and columns to reconstitute original values

        :param M : matrix to normalize
        :param n : number of non-zero values in the matrix
        :param zero_values : where the zero values are located in M
        :return: a vector containing the RMSE for each row
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
    
        normalized_m[zero_values] = 0

        return normalized_m, avg_users, avg_items

        

