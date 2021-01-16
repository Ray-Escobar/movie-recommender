import numpy as np
import math
from matrix_factorization.UvDecomposition import UvDecomposer

import sys
sys.path.append('.')

from commons.FormulaFactory import FormulaFactory
from commons.FormulaFactory import ScoringMeasureType
from commons.PredictionStrategy import PredictionStrategy
from data_handling.DataLoader import DataLoader

class SimpleUVDecomposer(UvDecomposer):

    """
    Makes predictions by performing an UV decomposition on the provided matrix
    """

    def __init__(self, d:int, iterations:int, mu:float, formula_factory:FormulaFactory, scorer_type: ScoringMeasureType):


        super().__init__(d, iterations, mu, formula_factory, scorer_type)
        # d cant be 1 or less
        assert(d > 1)

        self.test = True
        
    def perform_precomputations(self):
        super().perform_precomputations()


    def decompose_matrices(self, row:int, col:int):
        if (self.test):
            print("in the simple UV decomposer :)")
            self.test = False

        gradient = 2*(self.M[row][col] - np.dot(self.U[row], self.V[:,col]))
        self.U[row]   = self.U[row]   + self.mu*(gradient*self.V[:,col])
        self.V[:,col] = self.V[:,col] + self.mu*(gradient*self.U[row])


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

        self.U[row][col] = self.mu*(numer / denom)

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
        
        self.V[row][col] = self.mu*(numer / denom)

