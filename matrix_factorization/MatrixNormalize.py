import numpy as np
import sys 

def normalize(M:np.array, zero_values:np.array) -> (np.array):
        """

        Normalizes a given matrix and returns a new normalized matrix with
        the average of rows and columns to reconstitute original values

        :param M : matrix to normalize
        :param n : number of non-zero values in the matrix
        :param zero_values : where the zero values are located in M
        :return: a vector containing the RMSE for each row
        """

        #np.set_printoptions(threshold=sys.maxsize)

        #boolean matrix to get sum of non-zero values 
        # to calcualte the averages
        non_zeros = M > 0 
        
        #print(non_zeros)
        users_i = non_zeros.sum(axis=1)
        items_j = non_zeros.sum(axis=0)


        avg_users = np.divide(M.sum(axis=1), users_i, where= users_i!=0) #if denom is 0, just assign 0
        avg_items = np.divide(M.sum(axis=0), items_j, where= items_j!=0) #if denom is 0, just assign 0



        normalized_m = np.zeros((len(M), len(M[0])))
        #now fill up M matrix with normalized values
        for row in range(len(M)):
            for col in range(len(M[0])):
                #substact entry m_ij by (avg_user+avg_itme)/2
                normalized_m[row][col] = M[row][col] - (avg_users[row]+avg_items[col])/2
    
        normalized_m[zero_values] = 0

        return normalized_m, avg_users, avg_items

        

''' Emergency code
def __decompose_matrix_u(row:int, col:int):
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

def __decompose_matrix_v(row:int, col:int):
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
    
    self.V[row][col] = mu*(numer / denom)
'''