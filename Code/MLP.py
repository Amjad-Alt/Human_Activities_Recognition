# import packages
import pandas as pd
import numpy as np
import os

# modeling 
from sklearn.neural_network import MLPClassifier # model chosen
from sklearn.model_selection import train_test_split 
#from bayes_opt import BayesianOptimization # bysian for alpha 
#from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# plotting
import matplotlib.pyplot as plt

# model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve


#=========================== LOAD DATA =========================

# set seed
np.random.seed(42)

# load dataset and saperate X,y
sounds = pd.read_csv("data/sounds.csv")

X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]

# Understand data such as NA valuse and do graphs and data type

# X,y shapes
print(f'X schema {X.shape}')
print(f'y schema {y.shape}')

# X,y head
print(X.head(3))
print(y.head(3))

# check features types
print(f'{X.dtypes}')
print(f'{y.dtypes}')

# make a copy of the main dataset 
sounds2 = sounds.copy()

# change y into numarical for the model
print(f'Activities before numarically label them {sounds2.iloc[:,0].unique()}')
y = sounds2.replace({'STANDING': 1, 'SITTING': 2, 'LAYING': 3, 'WALKING': 4, 'WALKING_DOWNSTAIRS':5, 'WALKING_UPSTAIRS':6})
print(f'Activities values after labeling {sounds2.iloc[:,0].unique()}')

#============================== Modeling ==========================

# split dataset into training, testing sets 60-40
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the hyperparameters search space
hyperparameters_space = {
    'hidden_layer_sizes': Integer(10, 100),
    'activation': Categorical(['relu', 'tanh']),
    'solver': Categorical(['adam', 'lbfgs']),
    'alpha': Real(1e-5, 1e-3, prior='log-uniform'),
    'learning_rate_init': Real(0.0001, 0.1, prior='log-uniform')
    }

# hidden_layer_sizes: the number of neurons in each hidden layer.
# activation: the activation function for the hidden layer neurons.
# solver: the optimization algorithm used to find the weights and biases of the neural network.
# alpha: L2 regularization parameter that helps to prevent overfitting.
# learning_rate_init: the initial learning rate used by the optimizer.

# 'relu': Rectified Linear Unit activation function
# 'tanh': Hyperbolic Tangent activation function
# 'adam': Adaptive Moment Estimation optimizer
# 'lbfgs': Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimizer

# Create the MLP classifier
mlp = MLPClassifier(random_state=42)

# Create the BayesSearchCV object with cross-validation
search = BayesSearchCV(mlp, hyperparameters_space, n_iter=50, cv=5, random_state=42)
# n_iter: number of iterations performed in the random search process

# fit the BayesSearchCV object to the training set
search.fit(X_train, y_train)

# Print the best hyperparameters found
print(search.best_params_)
# OrderedDict([('activation', 'tanh'), ('alpha', 0.001), ('hidden_layer_sizes', 50), ('learning_rate_init', 0.0012222687915984045), ('solver', 'adam')])
print(search.best_score_)

# ============================== Evaluation =========================
# accurcy
best_model = search.best_estimator_
accuracy = cross_val_score(best_model, X_valtest, y_valtest, cv=5).mean()
print("Accuracy:", accuracy)
# Accuracy: 0.9781553398058251

# confusion matrix
# predict classes of the test set
y_pred = best_model.predict(X_valtest) 

# Calculate confusion matrix of test
cm = confusion_matrix(y_valtest, y_pred) 
print("Confusion Matrix:")
print(cm)

# Calculate precision, recall, and f1-score
print(classification_report(y_valtest, y_pred)) 

plot_confusion_matrix(best_model, X_valtest, y_valtest) 
plt.title("Confusion Matrix")
plt.show()

# learning curve
# Define range of sample sizes to plot learning curve
train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Calculate learning curve scores for both training and test 
train_sizes, train_scores, val_scores = learning_curve(best_model, X_train, y_train, train_sizes=train_sizes, cv=5)

# Calculate mean and standard deviation of training scores and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score')
plt.plot(train_sizes, val_mean, label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy score')
plt.legend(loc='best')
plt.title('Learning Curve')
plt.show()



