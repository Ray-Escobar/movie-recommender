import unittest
import numpy as np

from commons.FormulaFactory import FormulaFactory, SimilarityMeasureType


class MyTestCase(unittest.TestCase):
    def test_cosine_similarity(self):
        u1 = np.array([0, -0.2, -0.2, 0, 1.8, -1.2, 0, 0,  1.8])
        u2 = np.array([0, 0, -1.16, 1.83, 0.83, -0.16, -2.16, 0, 0.83])

        formula_factory = FormulaFactory()
        sim = formula_factory.create_similarity_measure(SimilarityMeasureType.COSINE_SIMILARITY)

        self.assertAlmostEqual(0.36, sim(u1, u2), delta=0.1)

    def test_cosine_meanless_similarity(self):
        u1 = np.array([0, 3, 1, 0, 5, 2, 0, 0, 5])
        u2 = np.array([0, 0, 2, 5, 4, 3, 1, 0, 4])

        formula_factory = FormulaFactory()
        sim = formula_factory.create_similarity_measure(SimilarityMeasureType.MEANLESS_COSINE_SIMILARITY)

        self.assertAlmostEqual(0.489, sim(u1, u2), delta=0.1)

    def test_weighted_avg(self):
        values = [(0.39, 5), (0.27, 4)]
        formula_factory = FormulaFactory()
        avg = formula_factory.create_rating_average_weighted_by_similarity_function()

        self.assertAlmostEqual(4.59, avg(values), delta=0.1)


if __name__ == '__main__':
    unittest.main()
