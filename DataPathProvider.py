class DataPathProvider:
    def __init__(self, movies_path: str, predictions_path: str, ratings_path: str, submission_path: str, users_path: str):
        self.moviesPath = movies_path
        self.predictionsPath = predictions_path
        self.ratingsPath = ratings_path
        self.submissionPath = submission_path
        self.usersPath = users_path

    def get_movies_path(self):
        return self.moviesPath

    def get_predictions_path(self):
        return self.predictionsPath

    def get_ratings_path(self):
        return self.ratingsPath

    def get_submissions_path(self):
        return self.submissionPath

    def get_users_path(self):
        return self.usersPath