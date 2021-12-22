# Decision_Tree_Regression

I implemented the decision tree regression algorithm on Python. Unlike regular linear regression, this algorithm is used when the dataset is a curved line. The algorithm uses decision trees to generate multiple regression lines recursively. The training dataset is split into two parts in each iteration and a regression line is fit. The split is made at the best possible point to minimize the Mean Squared Error (MSE).

The number of regression lines is key. Overfitting occurs if the number is too high and underfitting occurs if the number is too low. There are two hyperparameters we use in this algorithm, maximum depth of the decision trees and the minimum number of samples in a single split. These parameters should be tested and optimized for each dataset.

# Creating Datasets

Instead of using datasets downloaded from the internet, I decided to create my own datasets for this project. I generated 4 datasets to test my algorithm: Noisy Sinusoidal Signal, Noisy Second Degree Polynomial, Noisy Linear Line and Noisy Upside Down Triangle Signal. The program generates these datasets when its run and saves the datasets to recreate the results. To generate new datasets, you simply need to delete the first dataset, dataset0.csv file. You can also use your own datasets by uploading them to the same directory as the Python project.

# Plotting Results

You can see the results of the sinusoidal signal and the upside down triangle for various hyperparameters. Colored points represent the splits in the training dataset, black lines represent the linear regression line for the corresponding split and the larger gray points represent the test dataset.

![Figure_1](https://user-images.githubusercontent.com/54302889/147139510-b197d21f-ad0d-443e-8da2-239378388a75.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147139727-08aba759-afdf-4235-8708-ec24d52dd1cd.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147139766-531a6b4e-3741-41f8-ae27-91fa7d1abc4d.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147139817-14cb0617-ac4b-4b9d-b67b-c946fbba85ab.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147139876-e97994f2-2b9c-4679-9a75-9c7729c4109e.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147139973-ab17afa2-831a-4ffb-9a54-384e2a398c63.png)

<br>

<br>

![Figure_1](https://user-images.githubusercontent.com/54302889/147140286-56d9f859-af39-4bd2-aa06-57cf03142b20.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147140312-723ad3be-c6fc-416d-8597-1dbe97fe1763.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147140332-2aae6bd1-cbd5-4a77-802f-9ee60c9f68c1.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147140346-b8099cab-4473-47a0-b7f3-e4f476e6d8be.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147140386-af90e2c6-e3f8-4895-afaf-75bc70d37a12.png)

![Figure_1](https://user-images.githubusercontent.com/54302889/147140406-24614a96-126b-486b-8d29-4eede1934047.png)

<br>

It is observed that for these datasets the best value for maximum depth is 4.
