# Report - Data Mining Project

## 1. Introduction

The aim of this project was to create a movie recommendation system
based on a dataset containing the ratings of 6040 netflix users for
3706 movies. 

Our training data consisted of 3 datasets, containing data about *movies*
, *users*, as well as user *ratings* for those movies. The movies dataset contained the
*year* and the *title* for each movie; the *users* dataset contained the *gender*, *age*, and a
number encoding their *profession*; most importantly, the *ratings* dataset
contained the ratings some *users* gave to certain *movies*.

What we, essentially tried to achieve was to make use of the available data  predict future ratings as accurately as possible. 
Our approach was based on 2 directions, according to which we split our team:

* Collaborative Filtering - Cătălin Lupău
* Latent Factor Decomposition - Pietro Vigilanza

## 2. Methodology

### Collaborative Filtering

#### LSH 

#### Naive

#### Global Baseline 

#### Agglomerative Clustering 

### Latent Factors

## 3. Results

### Collaborative Filtering

For Naive and GlobalBaseline: 
Weights: 0.3-0.7, 0.5-0.5, 0.7-0.3
Number of neighbors: 5, 15, 30, 50

LSH:
Weights: 0.3-0.7, 0.5-0.5, 0.7-0.3
Number of neighbors: 5, 15, 30, 50
Distance measure: cosine / pearson

Clustering:
Sizes: (30%, 30%), (50%, 50%), (70%, 70%)
Number of neighbors: 5, 15, 30, 50
Number of samples: 10, 100, 1000

We cannot test all possible combinations, but we can try a few
to compare the influence of each parameter:

Naive and GlobalBaseline:
(0.3-0.7, 30), (0.5, 0.5, 30), (0.7-0.3, 30)
(best, 5), (best, 15), (best, 30), (best, 50)

LSH:
(0.3-0.7, 30, cosine), (0.5, 0.5, 30, cosine), (0.7-0.3, 30, cosine)
(best, 5, cosine), (best, 15, cosine), (best, 30, cosine), (best, 50, cosine)
(best, best, cosine), (best, best, pearson)

Clustering:
((30%, 30%), 30, 100), ((50%, 50%), 30, 100), ((70%, 70%), 30, 100)
(best, 5, 100), (best, 15, 100), (best, 30, 100), (best, 50, 100)
(best, best, 10), (best, best, 100), (best, best, 1000)




#### LSH
Table with results for different parameters and weights.
Discussion on how the different parameters affect the results.

#### Naive 
Table with results for different parameters and weights.
Discussion on how the different parameters affect the results.

#### Clustering
Table with results for different parameters and weights.
Discussion on how the different parameters affect the results.

#### Global Baseline
Table with results for different parameters and weights.
Discussion on how the different parameters affect the results.

#### Comparison of collaborative filtering methods
Take the best result obtained by each method and compare it with the others.


### Latent Factors

### All Models Combined
Run a final algorithm using all models. Choose the best individual parameters. Justify the weights based on observed performance. 
Check it on kaggle.

## 4. Final Discussion and Conclusion

Lecture graph with how


