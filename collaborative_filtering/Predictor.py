class Predictor:
    """
    Interface implemented by classes responsible for predicting the value of the ratings matrix at a certain row and column
    """
    def predict(self, row: int, column: int):
        """
        Predict the rating at the provided row and column.

        :param row: the row to predict at
        :param column: the column to predict at
        :return: the rating, or 0 if no rating can be predicted
        """
        pass