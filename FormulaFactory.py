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
        return lambda similarity_rating_tuples: sum([sim * rating for sim, rating in similarity_rating_tuples]) / sum([sim for sim, _ in similarity_rating_tuples])

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
            non_zero_elements = [x for x in vec.tolist() if x != 0]
            mean = sum(non_zero_elements) / len(non_zero_elements)

            meanless_vec = []

            for e in vec.tolist():
                if e == 0:
                    meanless_vec.append(0)
                else:
                    meanless_vec.append(e - mean)
            return np.array(meanless_vec)


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