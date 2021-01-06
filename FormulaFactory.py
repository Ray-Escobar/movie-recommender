import numpy as np
from enum import Enum

class SimilarityMeasureType(Enum):
    COSINE_SIMILARITY = 1
    MEANLESS_COSINE_SIMILARITY = 2

class FormulaFactory:
    """
    Factory class used to create useful formulas to be used across the application
    """

    def create_rating_average_weighted_by_similarity_function(self):
        """
        Creates a function that computes the similarity weighted ratings average.

        :return: a function that performs the described action
        """

        def compute_weighted_average(tuples):
            weights, ratings = list(zip(*tuples))


            sm = np.sum(weights)

            if sm == 0:
                return 0.0

            return np.average(ratings, weights=weights)

        return lambda similarity_rating_tuples: compute_weighted_average(similarity_rating_tuples)

    def create_cosine_similarity_measure(self):
        """
        Creates a function that computes the cosine similarity measure between 2 vectors.

        :return: a function that performs the described action.
        """
        return lambda vec_x, vec_y: np.dot(vec_x, vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))

    def create_meanless_cosine_similarity_measure(self):
        """
        Creates a function that computes the cosine simialrity between 2 vectors after subtracting their mean.

        :return: a function the performs the described action.
        """
        def make_meanless(vec: np.array):
            vec = vec.astype('float64')
            non_zero_elements = vec[vec != 0]
            mean = np.mean(non_zero_elements)

            sub_mean = np.vectorize(lambda x: 0.0 if x == 0 else float(x) - mean)

            return sub_mean(vec)




        cosine_similarity = self.create_cosine_similarity_measure()

        return lambda vec_x, vec_y: cosine_similarity(make_meanless(vec_x), make_meanless(vec_y))

    def create_similarity_measure(self, similarity_measure_type: SimilarityMeasureType):
        """
        Returns a function the performs the desired similarity type
        :param similarity_measure_type: the similarity measure to be created
        :raises Exception: if the provided type is unsuported.

        :return: a similarity measure of the desired type
        """
        if similarity_measure_type is SimilarityMeasureType.COSINE_SIMILARITY:
            return self.create_cosine_similarity_measure()

        elif similarity_measure_type is SimilarityMeasureType.MEANLESS_COSINE_SIMILARITY:
            return self.create_meanless_cosine_similarity_measure()

        else:
            raise Exception("Unsuported similarity type!")