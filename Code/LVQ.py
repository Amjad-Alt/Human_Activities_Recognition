
# ============================= MODELING ================================
# split dataset into training, testing sets 60-40

# define parameter distributions for random search
param_dist = {'prototypes_per_class': range(1, 11),
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
# 10 PCA Best score: 0.8297458171193479
# Best score: 0.9145518329884874
print(f'Best parameters:{random_search.best_params_}')
# Best parameters:{'prototypes_per_class': 7, 'beta': 75}
# 10 PCA Best parameters:{'prototypes_per_class': 7, 'beta': 75}
# 20 PCA Best parameters:{'prototypes_per_class': 3, 'beta': 75}
#============================= Evaluation ===========================

# accurcy
best_model = random_search.best_estimator_
accuracy = cross_val_score(best_model, X_test, y_test, cv=5).mean()
print("Accuracy:", accuracy)
# Accuracy: 0.9769417475728155
# 10 PCA Accuracy: 0.8208737864077669
# 20 PCA Accuracy: 0.8927184466019418

# confusion matrix
# predict the classes of the test set
y_pred = best_model.predict(X_test) 

# Calculate confusion matrix of test set
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:")
print(cm)

# Calculate precision, recall, and f1-score
print(classification_report(y_test, y_pred)) 

plot_confusion_matrix(best_model, X_test, y_test) 
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
som_width = int(20)
som_height = int(20)
som = MiniSom(x=10, y=10, input_len=20, sigma=1.0, learning_rate=0.5)
som.pca_weights_init(X_train.values)
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




