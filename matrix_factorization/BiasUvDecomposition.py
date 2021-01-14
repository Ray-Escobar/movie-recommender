import numpy as np
import sys
sys.path.append('.')

from FormulaFactory import FormulaFactory, ScoringMeasureType
from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
from matrix_factorization import MatrixNormalize
from matrix_factorization.UvDecomposition import UvDecomposer

class BiasUvDecomposer(UvDecomposer):

    """
    Makes predictions by performing an UV decomposition on the provided matrix
    """

    def __init__(self, d:int, iterations:int, mu:float, delta1:float, delta2:float, formula_factory:FormulaFactory, 
                       scorer_type: ScoringMeasureType, bias_weight:int):


        super().__init__(d, iterations, mu, formula_factory, scorer_type)
        # d cant be 1 or less
        assert(d > 1)
        self.delta1 = delta1
        self.delta2 = delta2
        self.bias_weight = bias_weight

        self.wow = True

        
        
    def perform_precomputations(self):
        super().perform_precomputations()

        self.mean_rating = self.M.sum()/self.number_of_values
        self.user_bias   = np.random.rand(len(self.M))
        self.movie_bias  = np.random.rand(len(self.M[0]))
        print("Bias UV Decomposer")

    def decompose_matrices(self, row:int, col:int):

        if (self.wow):
            print("Starting with bias UV")
            self.wow = False

        gradient = 2*(self.M[row][col]-(self.mean_rating + self.user_bias[row] + self.movie_bias[col] + np.dot(self.U[row], self.V[:,col]) ))
        
        self.U[row]   = self.U[row]   + self.mu*(gradient*self.V[:,col] - self.delta1 * self.U[row]  )
        self.V[:,col] = self.V[:,col] + self.mu*(gradient*self.U[row]   - self.delta2 * self.V[:,col])

        ### now altering the biases

        self.user_bias[row]  = self.user_bias[row]  + self.mu*(gradient - self.bias_weight*self.user_bias[row] )
        self.movie_bias[col] = self.movie_bias[col] + self.mu*(gradient - self.bias_weight*self.movie_bias[col] )


    



