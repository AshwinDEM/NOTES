
# Artificial Intelligence
## Unit 1







# Machine Learning
## Unit 3

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at task in T as measured by P improves with experience E.

The types of Machine Learning are:
1. Supervised Learning
	1. Classification
	2. Regression
2. Unsupervised Learning
	1. Clustering
	2. Association Analysis
3. Reinforcement Learning

##### Comparison of Various Machine Learning Techniques

| Supervised                                             | Unsupervised                                                           | Reinforcement                                                  |
| ------------------------------------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------- |
| Used for Classifying                                   | To find pattern                                                        | Rewards - if guessed correctly                                 |
| Labelled training data                                 | Unknown and unlabeled data                                             | Model learns and updates itself through rewards and punishment |
| Classification and Regression                          | Clustering and Association analysis                                    | No such types                                                  |
| Naive Bayes, KNN, LR, SVM                              | Kmeans, PCA, DBSCAN, apriori                                           | Q-Learning, Sarsa                                              |
| Handwriting Recognition, stock market prediction, etc. | Market based analysis,<br>recommender system, customer<br>segmentation | Self driving cars, intelligent robots                          |
| Simple to understand                                   | More difficult to understand and implement                             | Most complex                                                   |

#### Descriptive and Predictive Modeling

**Descriptive**: Determines similarities in the data and to find existing patterns.
Ex: Clustering, Association, Anomaly detection

Predictive: It uses the supervised learning functions which are used to
predict the target value.
Ex: Regression, Decision Trees, and NN

#### Types of Data Attributes

1. Qualitative
	1. Nominal: Variables with no inherent order or ranking. Ex: Blood group, gender, race.
	2. Ordinal: Variables with ordered series. Ex: Low/Middle/High income
	3. Binary: Only 2 possible values. Ex: Pass/Fail, Yes/No
2. Quantitative:
	1. Discrete: Based on counts. Ex: Number of students in a class
	2. Continuous: Can be measured on a continuum or a scale. Ex: Length, height



#### Underfitting vs Overfitting

| To avoid Underfitting (High Bias) | To avoid Overfitting (High Variance) |
| --------------------------------- | ------------------------------------ |
| Increase model complexity         | Increase the training data           |
| Train for longer                  | Less complex model                   |
| Increase the number of features   | Remove less relevant features        |
| Decrease regilarization           | L1/L2 regularization                 |
| Choose a difference model         | Ensembling                           |

#### Evaluating Performance

Correlation Matrix

| True Positive  | False Positive |
| -------------- | -------------- |
| False Negative | True Negative  

$$
Accuracy = \frac{TP + TN}{TP + FP + FN + TN}
$$
Sensitivity measures the proportion of positive cases that were correctly classified
Specificity measures the proportion of negative cases that were correctly classified

$$
Sensitivity = \frac{TP}{TP + FN}
$$

$$
Specificity = \frac{TN}{TN + FP}
$$

Precision measures the proportion of positive identification that are correct
$$ 
Precision = \frac{TP}{TP + FP}
$$

Recall measures the number of actual positives that were correctly identified
$$
Recall = \frac{TP}{TP + FN}
$$

F-Measure is the harmonic mean of precision and recall

$$
F-Measure = \frac{2 * Precision * Recall}{Precision + Recall}
$$
Kappa value of a model indicated the adjusted model accuracy
$$
k = \frac{Observed Agreement - Expected Agreement}{1 - Expected Agreement}
$$
where Observed Agreement is the Accuracy and 

$$
Expected Agreement = \frac{(TP * FP + TP * FN) + (TN * FN + TN * FP) }{TP + FP + FN + TN}
$$