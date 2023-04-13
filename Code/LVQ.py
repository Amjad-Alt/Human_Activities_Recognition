
# import packages
import pandas as pd
import numpy as np

# plotting
import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot
from minisom import MiniSom

# model
from sklearn_lvq import GlvqModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
#from skopt.space import Real, Integer, Categorical

# evaluation
#from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve

# --------------------------- PREPARE DATA ----------------------------

# set seed
np.random.seed(42) 

# load dataset and saperate X,y
sounds = pd.read_csv("sounds.csv", skiprows=[0], header=None)

X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]

# change y into numarical
print(f'y values before numarically label them {y.iloc[:,0].unique()}')
y = y.replace({'STANDING': 1, 'SITTING': 2, 'LAYING': 3, 'WALKING': 4, 'WALKING_DOWNSTAIRS':5, 'WALKING_UPSTAIRS':6})
print(f'y values after labeling {y.iloc[:,0].unique()}')

# ----------------------------- MODELING ------------------------------
# split dataset into training, testing sets 60-40

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.4, random_state=42)


# define parameter distributions for random search
param_dist = {'prototypes_per_class': range(1, 11),
              'random_state': range(10),
              'beta': [0, 25, 5, 75, 1]
              }
#beta: controls the learning rate for the prototypes.
#c: controls the weighting given to misclassified samples during training.
#display: controls the amount of information displayed during training.
#gtol: controls the tolerance for the optimization algorithm used during training.
#initial_prototypes: controls the initial positions of the prototypes.
#prototypes_per_class: controls the number of prototypes per class.
#random_state: sets the random seed for reproducibility.

# create an instance of the LVQ model
lvq_model = GlvqModel()

# create a random search object
random_search = RandomizedSearchCV(estimator=lvq_model, param_distributions=param_dist, 
                                   n_iter=10, cv=5, verbose=1)
# verbose=1: randomized progress of the search will be printed to the console during the search
# n_iter=10: randomized search will sample 10 different combinations of hyperparameters

# fit random search object to data
random_search.fit(X_train, y_train.values.ravel())

# print best parameters and corresponding score
print(f'Best score: {random_search.best_score_}')
# Best score: 0.9721638300381275
print(f'Best parameters:{random_search.best_params_}')
# Best parameters:{'random_state': 8, 'prototypes_per_class': 7, 'beta': 75}

#----------------------------- EVALUATION ----------------------------

# accurcy
best_model = random_search.best_estimator_
accuracy = cross_val_score(best_model, X_valtest, y_valtest, cv=5).mean()
print("Accuracy:", accuracy)
# Accuracy: 0.9769417475728155

# confusion matrix
# predict the classes of the test set
y_pred = best_model.predict(X_valtest) 

# Calculate confusion matrix of test set
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


# plot SOM
# Get the trained prototypes from the LVQ model
prototypes = best_model.w_

# Use the SOM algorithm to create a self-organizing map using the prototypes as input
som_width = 10
som_height = 10
som = MiniSom(som_width, som_height, X_train.shape[1], sigma=1.0, learning_rate=0.5)
som.pca_weights_init(X_train)
som.train_batch(prototypes, 10000, verbose=True)

# Plot the SOM to visualize the clustering regions
plt.figure(figsize=(10, 10))
pcolor(som.distance_map().T, cmap='bone_r') 
colorbar()
markers = ['o', 's', 'D', 'v', '^', 'p']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
for i, x in enumerate(X_train):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y_train.values.ravel()[i]-1], 
         markerfacecolor='None', markeredgecolor=colors[y_train.values.ravel()[i]-1], markersize=10, markeredgewidth=2)
plt.show()



