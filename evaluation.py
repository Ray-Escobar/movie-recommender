from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.ClusterCollaborativeFiltering import ClusterCollaborativeFiltering
from collaborative_filtering.global_baseline.ItemGlobalBaselineCollaborativeFiltering import \
    ItemGlobalBaselineCollaborativeFiltering
from collaborative_filtering.global_baseline.UserGlobalBaselineCollaborativeFiltering import \
    UserGlobalBaselineCollaborativeFiltering
from collaborative_filtering.naive.ItemNaiveCollaborativeFiltering import ItemNaiveCollaborativeFiltering
from collaborative_filtering.naive.UserNaiveCollaborativeFiltering import UserNaiveCollaborativeFiltering
from commons.FormulaFactory import FormulaFactory
from commons.RatingPredictor import RatingPredictor
from data_handling.CsvProvider import CsvProvider
from data_handling.DataLoader import DataLoader
from data_handling.DataPathProvider import DataPathProvider
from data_handling.DiskPersistor import DiskPersistor
from data_handling.LocalFileCsvProvider import LocalFileCsvProvider
from evaluation_tools import generate_prediction_report

#####
##
## DATA IMPORT
##
#####

# Where data is located

movies_file = './data/evalutation/reduced_movies.csv'
users_file = './data/evalutation/reduced_users.csv'
ratings_file = './data/evalutation/ratings_train.csv'
predictions_file = './data/evalutation/predictions_test.csv'
expected_ratings_file = './data/evalutation/expected_predictions_test.csv'
submission_file = 'data/submissions/submission.csv'

# Create a data path provider
data_path_provider = DataPathProvider(movies_path=movies_file, users_path=users_file, ratings_path=ratings_file, predictions_path=predictions_file, submission_path=submission_file)

# Creata a data loader
data_loader = DataLoader(data_path_provider=data_path_provider, csv_provider=LocalFileCsvProvider())
disk_persistor = DiskPersistor()
formula_factory = FormulaFactory()

# load the expected ratings
csv_provider: CsvProvider = LocalFileCsvProvider()
expected_ratings_array = csv_provider.read_csv(expected_ratings_file, delimiter=';', column_names=['userID', 'movieID', 'rating']).to_numpy()
# convert to a dictionary
expected_ratings_dict = dict()

for instance in expected_ratings_array:
    user_id, movie_id, expected_rating = instance[0], instance[1], instance[2]
    expected_ratings_dict[(user_id, movie_id)] = expected_rating

print(data_loader.get_ratings_matrix())

# Create the user similarity matrix matrix if not already created
sym_matrix_results = disk_persistor.perist_computation([
    (lambda: RowPearsonSimilarityMatrix(data_loader.get_ratings_matrix()), 'evaluation_global_pearson_similarity_matrix'),
    (lambda: RowPearsonSimilarityMatrix(data_loader.get_ratings_matrix().T), 'evaluation_global_pearson_similarity_matrix_movie')
], force_update=True)

global_pearson_similarity_matrix_user = sym_matrix_results[0]
global_pearson_similarity_matrix_movie = sym_matrix_results[1]


#####
##
## ACTUAL EVALUTATION
##
#####

evaluation_naive = {
    'name': 'Naive Collaborative Filtering',
    'description': 'Naive Collaborative Filtering with weights 0.7 and 0.3 item - user',
    'weights': [0.7, 0.3],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_predictor_naive',
                        prediction_strategies=[
                                ItemNaiveCollaborativeFiltering(
                                    k_neighbors=30,
                                    sim_matrix=global_pearson_similarity_matrix_movie
                                ),
                                UserNaiveCollaborativeFiltering(
                                    k_neighbors=30,
                                    sim_matrix=global_pearson_similarity_matrix_user
                                 )
                        ]
                    )

}

evaluation_clustering = {
    'name': 'Clustring Based Collaborative Filtering',
    'description': 'Clustring based collaborative Filtering with a 20% compression rate for both rows and columns',
    'weights': [1.0],
    'force_update': True,
    'predictor': RatingPredictor(
                    data_loader=data_loader,
                    disk_persistor=disk_persistor,
                    persistence_id='predictor_clustering',
                    prediction_strategies=[
                        ClusterCollaborativeFiltering(
                            row_similarity_matrix=global_pearson_similarity_matrix_user,
                            col_similarity_matrix=global_pearson_similarity_matrix_movie,
                            new_dim_ratio=(0.8, 0.8),
                            k_neighbors=35,
                            randomized=True,
                            randomized_num_extractions=100,
                            random_seed=3
                        )
                    ]
                )

}

global_biases = {
    'name': 'Global Biases Collaborative Filtering',
    'description': 'Global Biases Collaborative Filtering with weights 0.7 and 0.3 item - user',
    'weights': [0.7, 0.3],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_predictor_naive',
                        prediction_strategies=[
                                ItemGlobalBaselineCollaborativeFiltering(
                                    k_neighbors=30,
                                    sim_matrix=global_pearson_similarity_matrix_movie
                                ),
                                UserGlobalBaselineCollaborativeFiltering(
                                    k_neighbors=30,
                                    sim_matrix=global_pearson_similarity_matrix_user
                                 )
                        ]
                    )

}


# Generate reports for evaluations

generate_prediction_report('test_report', [evaluation_naive, evaluation_clustering, global_biases], expected_ratings_dict)


