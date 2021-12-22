# Decision_Tree_Regression

I implemented the decision tree regression algorithm on Python. Unlike regular linear regression, this algorithm is used when the dataset is a curved line. The algorithm uses decision trees to generate multiple regression lines recursively. The training dataset is split into two parts in each iteration and a regression line is fit. The split is made at the best possible point to minimize the Mean Squared Error (MSE).

The number of regression lines is key. Overfitting occurs if the number is too high and underfitting occurs if the number is too low. There are two hyperparameters we use in this algorithm, maximum depth of the decision trees and the minimum number of samples in a single split. These parameters should be tested and optimized for each dataset.

# Creating Datasets

Instead of using datasets downloaded from the internet, I decided to create my own datasets for this project. I generated 4 datasets to test my algorithm: Noisy Sinusoidal Signal, Noisy Second Degree Polynomial, Noisy Linear Line and Noisy Upside Down Triangle Signal. The program generates these datasets when its run and saves the datasets to recreate the results. To generate new datasets, you simply need to delete the first dataset, dataset0.csv file. You can also use your own datasets by uploading them to the same directory as the Python project.

# Plotting Results
