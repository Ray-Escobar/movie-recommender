from typing import List

import numpy as np
import json

from commons.RatingPredictor import RatingPredictor

EVALUATION_REPORTS_DIR = './evaluation_reports'


def rmse(predicted: dict, expected: dict) -> float:
    """
    Computes the root mean squared error between the expected and the predicted values of the 2 provided dictionaries.
    :param predicted: the dictionary containing the predicted values
    :param expected: the dictionary containing the expected values
    :return: the root mean squared error between the 2 values
    """

    error = 0

    for key in predicted.keys():
        pred_rat = predicted[key]
        exp_rat = expected[key]
        error += (pred_rat - exp_rat) ** 2


    if len(predicted.keys()) == 0:
        return 0.0

    error = error / len(predicted.keys())

    return np.sqrt(error)



def evaluate_predictors(predictor_descriptors: List[dict], expected_ratings: dict) -> List[float]:
    """
    Computes the RMSE of the provided predictors.

    :param predictor_descriptors: - a list of python dictionaries containing RatingPredictors,
        and other related data such as weights, whether or not perform a force update
    :param expected_ratings: - the dictionary containing the expected ratings for the predictors
    :return: a list containing the rmse of each predictor
    """

    scores = []

    for predictor_descriptor in predictor_descriptors:
        predictor: RatingPredictor = predictor_descriptor['predictor']
        weights: List[float] = predictor_descriptor['weights']
        force_update: bool = predictor_descriptor['force_update']

        predictor.perform_precomputations(force_update)
        predicted_ratings = predictor.make_average_prediction(weights)

        score = rmse(predicted_ratings, expected_ratings)

        scores.append(score)

    return scores

def generate_prediction_report(report_name: str, predictor_descriptors: List[dict], expected_ratings: dict):
    """
    Generates an evaluation report with the score of each predictor.

    :param report_name: the name of the report.
    :param predictor_descriptors: a list of python dictionaries containing RatingPredictors,
        and other related data such as weights, whether or not perform a force update, as well as a name and
        a description for the predictor.
    :param expected_ratings: a dictionary containing the expected ratings for the predictors.
    """

    scores: List[float] = evaluate_predictors(predictor_descriptors, expected_ratings)

    with open('{}/{}.json'.format(EVALUATION_REPORTS_DIR, report_name), 'w') as report:

        for i, descriptor in enumerate(predictor_descriptors):
            score = scores[i]

            description = {
                'Predictor name': descriptor['name'],
                'Description': descriptor['description'],
                'Estimated RMSE Score': score
            }
            report.write(json.dumps(description, indent=4))



