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

For our approach of latent factors, we applied UV decompostion of our rating matrix M. In more detail, this algorithm creats matrices U and V which try to approximate the original matrix M the best possible. As such, we try to achieve the equality below as close as possible:

$$
M = UV
$$


For this method, there isn't an algebraic approach to finding the optimal UV matrix, but rather we try to get as close to an optimal solution by the gradinet descent techique.

The three implementaions discussed below have some constants in the way they are implemented. The constants in the implementations are teh following:

- The U matrix represents the "people ratings concept" (the long matrix)
- The V matrix represents the "movie ratings concept"  (the wide matrix)
- Rating matrix M has n x m dimensions. U matrix has n x d and V matrix has d x m dimensions
    - the d is also a parameter we can adjust in all of our models
- The matrices U and V are initialized with thier entries being $\sqrt(avg/d) + random_value$.  The random value nudges this preset value either to the negative or the positive side. This adds some randomness to the convergence procedure.
- All implementation have a learning rate $\mu$
- All implementations use RMSE to judge the progress of the gradient descnet process
- Only the stochastic gradeint descent process is implement. SGD processes are generally seen as better and more realistic to implement in big data sets
    - Our SGD implementation works by altering the row from U and column from V that affects the score $r_xi$ of the matrix
    - For SGD we only alter U and V for existing values of the rating matrix M, hence we ignore all values that are 0 in our RMSE SGD process. 

#### Original UV Decompositon

The intial implementation consisted of simply reducing RMSE without considering any regulatization terms. To do this, we simply iterated over each term in the matrix and calucalted the gradient of the simple function. This implementation followed closely this equation from the book in page {add page}:

In this step we also tried to normalize the original M matrix, but we noticed the gradient was
very unstable with these smaller values. This mean that the RMSE would never converge to a value
and sometimes even some values tended towards infinty. 

This original RMSE implementation was decent, but it was affected by overfitting. This is becasue without regularization terms, the values of the U and V matrices are free to expand without and grow without being held back. THis meant that the model would easily overfit and not be able to generalize.

#### Regularized UV Decompositon

Regularized UV compostion turned out to be the a huge step into the right direction. With some simple parameter tuning we were able to easily beat the results of our collaborative filtering algorithms.

The main difference with this approach is that we added regularization terms, which makes gradeint adjustments be more conservatiVe. Hence this model contained two extra parameters - namely delta_1 and delta_2. delta_1 corresponded to the regularization term for the rows of the original matrix (the users), while delta2 consisted of the regularization term for the columns of the original matrix (the movies). The equation is the one below:

The regularization terms were crucial and lead towards investigating new approaches to improve even more.

#### Biases-Regularized UV Decomposition

The Biased-Regularized UV decomposer is our most complex model for latent factors since it included the most amount of parameters to tune. For this example we added a bias for the user and the movie. This meant our model was able to spot certain biases from user - like someone who generaly rates a movie highly, or a movie that is generally hated by all audiences. 

The function we minimized was the following:

As such, this algorithm was good since it establishes a baseline with the mean and biases, and simply adjusts its weights to to approach the correct values for each entry. This approach worked better since it took knwoledge and input from the real world.

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

For Simple UV Decomposer: 
Mu = 0.005 - 0.003
Number of neighbors: 5, 15, 30, 50

LSH:
Weights: 0.3-0.7, 0.5-0.5, 0.7-0.3
Number of neighbors: 5, 15, 30, 50
Distance measure: cosine / pearson

Clustering:
Sizes: (30%, 30%), (50%, 50%), (70%, 70%)
Number of neighbors: 5, 15, 30, 50
Number of samples: 10, 100, 1000


#### Simple UV decomposer

#### Regularized UV decomposer

#### Biased-Regularized UV decomposer

### All Models Combined
Run a final algorithm using all models. Choose the best individual parameters. Justify the weights based on observed performance. 
Check it on kaggle.

## 4. Final Discussion and Conclusion

Lecture graph with how


