from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.clustering.ClusterCollaborativeFiltering import ClusterCollaborativeFiltering
from collaborative_filtering.global_baseline.ItemGlobalBaselineCollaborativeFiltering import \
    ItemGlobalBaselineCollaborativeFiltering
from collaborative_filtering.global_baseline.UserGlobalBaselineCollaborativeFiltering import \
    UserGlobalBaselineCollaborativeFiltering
from collaborative_filtering.lsh.ItemLshCollaborativeFiltering import ItemLshCollaborativeFiltering
from collaborative_filtering.lsh.UserLshCollaborativeFiltering import UserLshCollaborativeFiltering
from collaborative_filtering.naive.ItemNaiveCollaborativeFiltering import ItemNaiveCollaborativeFiltering
from collaborative_filtering.naive.UserNaiveCollaborativeFiltering import UserNaiveCollaborativeFiltering
from commons.FormulaFactory import FormulaFactory, SimilarityMeasureType
from commons.RatingPredictor import RatingPredictor
from data_handling.CsvProvider import CsvProvider
from data_handling.DataLoader import DataLoader
from data_handling.DataPathProvider import DataPathProvider
from data_handling.DiskPersistor import DiskPersistor
from data_handling.LocalFileCsvProvider import LocalFileCsvProvider
from evaluation_tools import generate_prediction_report
from commons.FormulaFactory import FormulaFactory
from commons.FormulaFactory import ScoringMeasureType
from matrix_factorization.RegularizedUvDecompositon import RegularizedUvDecomposer
from matrix_factorization.BiasUvDecomposition import BiasUvDecomposer
from matrix_factorization.SimpleUvDecomposition import SimpleUVDecomposer

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
], force_update=False)

global_pearson_similarity_matrix_user = sym_matrix_results[0]
global_pearson_similarity_matrix_movie = sym_matrix_results[1]


#Generate formular factory and True RMSE score
formula_factory = FormulaFactory()
scoring_measure_rmse = ScoringMeasureType.TRUE_RMSE
scoring_measure_bias = ScoringMeasureType.BIAS_TRUE_RMSE


#####
##
## ACTUAL EVALUTATION
##
#####
'''
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


regularized_UVdecomposer = {
    'name': 'Regularized UV Decomposer',
    'description': 'Regularized UV decomposer, D=50, mu = 0.005, delta1= 1.09, delta2 = 1.08',
    'weights': [1.0],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_uv_regularized',
                        prediction_strategies=[
                                RegularizedUvDecomposer(
                                    iterations=35,
                                    d=30,
                                    mu= 0.003,
                                    delta1=0.52,
                                    delta2=0.51,
                                    formula_factory = formula_factory,
                                    scorer_type=scoring_measure
                                )
                            ]
                    )

}

'''


simple_uv = {
    'name': ' biased one Regularized UV Decomposer',
    'description': 'Simple UV, D=7, mu = 0.003,',
    'weights': [1.0],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_uv_regularized',
                        prediction_strategies=[
                            SimpleUVDecomposer(
                                d = 7,
                                iterations=55,
                                mu=0.003,
                                formula_factory=formula_factory,
                                scorer_type=scoring_measure_rmse
                            )
                        ]
                    )

}


regularized_uv = {
    'name': 'Regularized UV Decomposer',
    'description': 'Regularized, D=7, mu = 0.003, delta1= 0.08, delta2 = 0.10',
    'weights': [1.0],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_uv_regularized',
                        prediction_strategies=[
                                RegularizedUvDecomposer(
                                    iterations=55,
                                    d=7,
                                    mu= 0.003,
                                    delta1=0.08,
                                    delta2=0.10,
                                    formula_factory = formula_factory,
                                    scorer_type=scoring_measure_rmse
                                )
                            ]
                    )

}


biased_uv = {
    'name': 'Biased UV Decomposer',
    'description': 'Biased 1, D=7, mu = 0.003, delta1= 0.08, delta2 = 0.10, bias1 = 0.07, bias2 = 0.09 ',
    'weights': [1.0],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_uv_regularized',
                        prediction_strategies=[
                                BiasUvDecomposer(
                                    iterations=55,
                                    d=7,
                                    mu= 0.003,
                                    delta1=0.08,
                                    delta2=0.10,
                                    bias_weight1=0.07,
                                    bias_weight2=0.09,
                                    formula_factory = formula_factory,
                                    scorer_type=scoring_measure_bias
                                )
                            ]
                    )

}

'''

biased3_UVdecomposer = {
    'name': 'Only Regularized UV Decomposer',
    'description': 'biased 3, D=50, mu = 0.005, delta1= 1.15, delta2 = 1.08',
    'weights': [0.40, 0.10, 0.5],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_uv_regularized',
                        prediction_strategies=[
                                ItemGlobalBaselineCollaborativeFiltering(
                                    k_neighbors=30,
                                    sim_matrix=global_pearson_similarity_matrix_movie
                                ),
                                UserGlobalBaselineCollaborativeFiltering(
                                    k_neighbors=30,
                                    sim_matrix=global_pearson_similarity_matrix_user
                                ),
                                BiasUvDecomposer(
                                    iterations=55,
                                    d=7,
                                    mu= 0.003,
                                    delta1=0.10,
                                    delta2=0.06,
                                    bias_weight1=0.11,
                                    bias_weight2=0.08,
                                    formula_factory = formula_factory,
                                    scorer_type=scoring_measure_bias
                                )
                            ]
                    )

}
'''


## Collaborative Filtering different parameters comparison

naive_collaborative_filtering = lambda w, k: {
    'name': 'Naive Collaborative Filtering',
    'description': 'Item-Item weight: {}, User-User weight: {}, k-neighbors: {}'.format(w, 1 - w, k),
    'weights': [w, 1 - w],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_predictor_naive',
                        prediction_strategies=[
                                ItemNaiveCollaborativeFiltering(
                                    k_neighbors=k,
                                    sim_matrix=global_pearson_similarity_matrix_movie
                                ),
                                UserNaiveCollaborativeFiltering(
                                    k_neighbors=k,
                                    sim_matrix=global_pearson_similarity_matrix_user
                                 )
                        ]
                    )

}

collaborative_filtering_with_global_biases = lambda w, k: {
    'name': 'Global Biases Collaborative Filtering',
    'description': 'Item-Item weight: {}, User-User weight: {}, k-neighbors: {}'.format(w, 1 - w, k),
    'weights': [w, 1 - w],
    'force_update': True,
    'predictor': RatingPredictor(
                        data_loader=data_loader,
                        disk_persistor=disk_persistor,
                        persistence_id='evaluation_predictor_naive',
                        prediction_strategies=[
                                ItemGlobalBaselineCollaborativeFiltering(
                                    k_neighbors=k,
                                    sim_matrix=global_pearson_similarity_matrix_movie
                                ),
                                UserGlobalBaselineCollaborativeFiltering(
                                    k_neighbors=k,
                                    sim_matrix=global_pearson_similarity_matrix_user
                                 )
                        ]
                    )

}

clustering_based_collaborative_filtering = lambda dim, k, sample_size: {
    'name': 'Clustring Based Collaborative Filtering',
    'description': 'Row and Column Dim Ratio: {}, k-neighbors: {}, sample-size: {}'.format(dim, k, sample_size),
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
                            new_dim_ratio=(dim, dim),
                            k_neighbors=k,
                            randomized=True,
                            randomized_num_extractions=sample_size,
                            random_seed=3
                        )
                    ]
                )

}

lsh_based_collaborative_filtering = lambda w, k, sign_len, max_query_distance, distance_measure: {
    'name': 'Cosine LSH based Collaborative Filtering',
    'description': 'Item-Item weight: {}, User-User weight: {}, k-neighbors: {}, Signiture Length: {}, Max Query Distance: {}, Distance Measure: {}'.format(w, 1 - w, k, sign_len, max_query_distance, distance_measure),
    'weights': [w, 1 - w],
    'force_update': True,
    'predictor': RatingPredictor(
                    data_loader=data_loader,
                    disk_persistor=disk_persistor,
                    persistence_id='predictor_lsh_based',
                    prediction_strategies=[
                        ItemLshCollaborativeFiltering(
                            k_neighbors=k,
                            signiture_length=sign_len,
                            max_query_distance=max_query_distance,
                            formula_factory=formula_factory,
                            random_seed=3,
                            cosine_similarity_type=SimilarityMeasureType.COSINE_SIMILARITY if distance_measure == 'cosine' else SimilarityMeasureType.MEANLESS_COSINE_SIMILARITY
                        ),
                        UserLshCollaborativeFiltering(
                            k_neighbors=k,
                            signiture_length=sign_len,
                            max_query_distance=max_query_distance,
                            formula_factory=formula_factory,
                            random_seed=3,
                            cosine_similarity_type=SimilarityMeasureType.COSINE_SIMILARITY if distance_measure == 'cosine' else SimilarityMeasureType.MEANLESS_COSINE_SIMILARITY
                        )
                    ]
    )
}




# Generate reports for evaluations

#generate_prediction_report('test_report', [evaluation_naive, evaluation_clustering, global_biases, regularized_UVdecomposer, biased_UVdecomposer], expected_ratings_dict)
#generate_prediction_report('graph_report', [ simple_uv, regularized_uv, biased_uv ], expected_ratings_dict)

## 20% test size, 40% data
# reports for naive collaborative clustering
# generate_prediction_report('naive_col_fil_weight_optimization',
#                            [
#                                naive_collaborative_filtering(0.3, 30),
#                                naive_collaborative_filtering(0.5, 30),
#                                naive_collaborative_filtering(0.7, 30)
#                            ],
#                            expected_ratings_dict) --> best weight: 0.5, RMSE: 0.9288482249041781
#
# generate_prediction_report('naive_col_fil_neighbors_optimization',
#                            [
#                                naive_collaborative_filtering(0.5, 5),
#                                naive_collaborative_filtering(0.5, 15),
#                                naive_collaborative_filtering(0.5, 30),
#                                naive_collaborative_filtering(0.5, 50)
#                            ],
#                            expected_ratings_dict)

# reports for global-biased collaborative clustering


# generate_prediction_report('global_bias_col_fil_weight_optimization',
#                            [
#                                 collaborative_filtering_with_global_biases(0.3, 30),
#                                 collaborative_filtering_with_global_biases(0.5, 30),
#                                 collaborative_filtering_with_global_biases(0.7, 30)
#                            ],
#                            expected_ratings_dict) --> best weight: 0.7, RMSE: 0.9038670758283112

# generate_prediction_report('global_bias_col_fil_neighbors_optimization',
#                            [
#                                collaborative_filtering_with_global_biases(0.7, 5),
#                                collaborative_filtering_with_global_biases(0.7, 15),
#                                collaborative_filtering_with_global_biases(0.7, 30),
#                                collaborative_filtering_with_global_biases(0.7, 50)
#                            ],
#                            expected_ratings_dict)

# reports for clustering based collaborative clustering

# generate_prediction_report('cluster_col_fil_mat_dim_optimization',
#                            [
#                                clustering_based_collaborative_filtering(0.3, 30, 100),
#                                clustering_based_collaborative_filtering(0.5, 30, 100),
#                                clustering_based_collaborative_filtering(0.7, 30, 100)
#                            ],
#                            expected_ratings_dict) -> best dim: 0.7, RMSE: 1.0818906585615655

# generate_prediction_report('cluster_col_fil_neighbors_optimization',
#                            [
#                                clustering_based_collaborative_filtering(0.7, 5, 100),
#                                clustering_based_collaborative_filtering(0.7, 15, 100),
#                                clustering_based_collaborative_filtering(0.7, 30, 100),
#                                clustering_based_collaborative_filtering(0.7, 50, 100)
#                            ],
#                            expected_ratings_dict)
#
# generate_prediction_report('cluster_col_fil_sample_optimization',
#                            [
#                                clustering_based_collaborative_filtering(0.7, 50, 10),
#                                clustering_based_collaborative_filtering(0.7, 50, 100),
#                                clustering_based_collaborative_filtering(0.7, 50, 1000),
#                                clustering_based_collaborative_filtering(0.7, 50, 4000)
#                            ],
#                            expected_ratings_dict)

# reports for LSH

generate_prediction_report('lsh_col_fil_optimize_sign_len',
                           [
                               lsh_based_collaborative_filtering(0.7, 30, 8, 5000, 'cosine'),
                               lsh_based_collaborative_filtering(0.7, 30, 16, 5000, 'cosine'),
                               lsh_based_collaborative_filtering(0.7, 30, 32, 5000, 'cosine'),
                           ],
                           expected_ratings_dict)


# generate_prediction_report('lsh_col_fil_optimize_dist_measure',
#                            [
#                                lsh_based_collaborative_filtering(0.7, 30, ?, 200, 'cosine'),
#                                lsh_based_collaborative_filtering(0.7, 30, ?, 200, 'pearson'),
#                            ],
#                            expected_ratings_dict)