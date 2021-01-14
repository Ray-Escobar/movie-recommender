import numpy as np
import sys
sys.path.append('.')

from FormulaFactory import FormulaFactory, ScoringMeasureType
from data_handling.DataLoader import DataLoader
from PredictionStrategy import PredictionStrategy
from matrix_factorization import MatrixNormalize
from matrix_factorization.UvDecomposition import UvDecomposer

class RegularizedUvDecomposer(UvDecomposer):

    """
    Makes predictions by performing an UV decomposition on the provided matrix
    """

    def __init__(self, d:int, iterations:int, mu:float, formula_factory:FormulaFactory, scorer_type: ScoringMeasureType,
                       delta1:int, delta2:int):


        super().__init__(d, iterations, mu, formula_factory, scorer_type)
        # d cant be 1 or less
        assert(d > 1)
        self.delta1 = delta1

        self.test = True
        
    def perform_precomputations(self):
        super().perform_precomputations()
        print("Only Regularized UV decomposer")


    def decompose_matrices(self, row:int, col:int):
        if (self.test):
            print("in the regularized one :)")
            self.test = False
        gradient = 2*(self.M[row][col] - np.dot(self.U[row], self.V[:,col]))
        self.U[row]   = self.U[row]   + self.mu*(gradient*self.V[:,col] - self.delta1 * self.U[row]  )
        self.V[:,col] = self.V[:,col] + self.mu*(gradient*self.U[row]   - self.delta1 * self.V[:,col])

