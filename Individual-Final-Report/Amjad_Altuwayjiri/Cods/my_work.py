# Packages needed

# standard packages
import pandas as pd
import numpy as np
import os

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import pcolor, colorbar, plot
from minisom import MiniSom

# clustering
from collections import Counter
from sklearn.cluster import DBSCAN 

# modeling 
from sklearn.neural_network import MLPClassifier # model chosen
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.decomposition import PCA
from sklearn_lvq import GlvqModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc


################# Do not run these codes it is one time thing ########################


#======================== Join train/test datasets =================================

# Set the working directory to the directory containing the datasets
os.chdir("C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/data")

# Create an empty list to store the dataframes
df_list = []

# Loop through the files in the directory
for file in os.listdir():
    # Check if the file is a CSV file
    if file.endswith(".csv"):
        # Read the CSV file into a pandas dataframe
       df = pd.read_csv(file)
        # Append the dataframe to the list
       df_list.append(df)

# Concatenate the dataframes in the list
concatenated_df = pd.concat(df_list, axis=0)

# save it as one file
concatenated_df.to_csv("sounds.csv", index=False)

# ================== Join raw data of single user ============================

# Define the directory where the files are stored
data_dir = "C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/data/HAPT_Data_Set/RawData"
save_dir = "C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/data"
# Loop users
for user in range(1, 31):
    
    # Define the filenames for this user's accelerometer and gyroscope data
    acc_filenames = [f"{data_dir}/acc_exp02_user01.txt", f"{data_dir}/acc_exp02_user01.txt"]
    gyro_filenames = [f"{data_dir}/gyro_exp01_user01.txt", f"{data_dir}/gyro_exp02_user01.txt"]
    
    # Read in the accelerometer and gyroscope data for this user
    acc_data = pd.concat([pd.read_csv(f, header=None, sep=" ") for f in acc_filenames])
    gyro_data = pd.concat([pd.read_csv(f, header=None, sep=" ") for f in gyro_filenames])
    
    # Join the accelerometer and gyroscope data horizontally
    data_h = pd.concat([acc_data, gyro_data], axis=1)

    # Add column names
    data_h.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Save the joined data to a file for this user
    data_h.to_csv(f"{save_dir}/user01_data.csv", index=False)
# =============================== Simple EDA ====================================
# set seed
np.random.seed(42)

# load dataset and saperate X,y
sounds = pd.read_csv("data/sounds.csv")

# data shapes
print(f'whole dataset schema {sounds.shape}')

# check Na values
print(f'number of Na in the whole dataframe {sounds.isna().sum().sum()}')

# data head
print(sounds.head(3))

# check features types
print(f'{sounds.dtypes}')

# get the frequency of each activity to check for balanced data
activity_counts = sounds['Activity'].value_counts()

# plot the bar chart 
plt.figure(figsize=(10,5))
plt.bar(activity_counts.index, activity_counts.values)
plt.title('Activity Frequency')
plt.xlabel('Activity')
plt.ylabel('Frequency')
plt.show()

# make a copy of the main dataset 
sounds2 = sounds.copy()

# change y into numarical for the model
sounds2['Activity'] = sounds2['Activity'].map({'STANDING': 1, 'SITTING': 2, 'LAYING': 3, 'WALKING': 4, 'WALKING_DOWNSTAIRS':5, 'WALKING_UPSTAIRS':6})


#=============================== Plot Gyroscope|Accelerometer ===========================

data = pd.read_csv('data/user01_data.csv')

palette = sns.color_palette("husl", 6)

# Create figure and axes objects
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 15))

# Plot 
ax1.plot(data.iloc[:,0], data.iloc[:,0], color=palette[0])
#ax1.set_title('accelerometer X')
ax2.plot(data.iloc[:,1].index, data.iloc[:,1], color=palette[1])
ax2.set_title('accelerometer Y')
ax3.plot(data.iloc[:,2].index, data.iloc[:,2], color=palette[2])
ax3.set_title('accelerometer Z')
ax4.plot(data.iloc[:,3].index,data.iloc[:,3], color=palette[3])
ax4.set_title('gyroscope X')
ax5.plot(data.iloc[:,4].index, data.iloc[:,4], color=palette[4])
ax5.set_title('gyroscope Y')
ax6.plot(data.iloc[:,5].index, data.iloc[:,5], color=palette[5])
ax6.set_title('gyroscope Z')

# Add figure title and legend
fig.suptitle('Subjcet1 Activity Gyroscope|Accelerometer', fontsize=14, fontweight='bold')
plt.show()

data2 = pd.read_csv('data/user02_data.csv')

# Plot 
ax1.plot(data2.iloc[:,0], data2.iloc[:,0], color=palette[0])
#ax1.set_title('accelerometer X')
ax2.plot(data2.iloc[:,1].index, data2.iloc[:,1], color=palette[1])
ax2.set_title('accelerometer Y')
ax3.plot(data2.iloc[:,2].index, data2.iloc[:,2], color=palette[2])
ax3.set_title('accelerometer Z')
ax4.plot(data2.iloc[:,3].index,data2.iloc[:,3], color=palette[3])
ax4.set_title('gyroscope X')
ax5.plot(data2.iloc[:,4].index, data2.iloc[:,4], color=palette[4])
ax5.set_title('gyroscope Y')
ax6.plot(data2.iloc[:,5].index, data2.iloc[:,5], color=palette[5])
ax6.set_title('gyroscope Z')

# Add figure title and legend
fig.suptitle('Subject2 Activity Gyroscope|Accelerometer', fontsize=14, fontweight='bold')
plt.show()

#=============================== Detect outliers ==================================

# Fit DBSCAN to the data
clustering = DBSCAN(eps=5, min_samples=2).fit(sounds2)
labels = clustering.labels_
# `eps` specifies maximum distance between two points in the same neighborhood
# `min_samples` specifies minimum number of points required to form a dense region


# Count the number of occurrences of each label
label_counts = dict(Counter(clustering.labels_))

# Print the number of clusters (excluding outliers -1)
num_clusters = len([k for k in label_counts if k != -1])
print(f"Number of clusters: {num_clusters}")
num_outliers = len([k for k in label_counts if k == -1])
print(f"Number of outliers: {num_outliers}")

# Get the unique labels and set colors for each cluster
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# Plot the data for each cluster
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    # Get the data points for this cluster
    xy = sounds2[class_member_mask & core_samples_mask]

    # Calculate the size of the marker based on the number of points in the cluster
    size = 100 * np.sqrt(np.count_nonzero(class_member_mask))

    # Plot the data for this cluster
    plt.plot(
        xy[0:1].mean(),
        xy[1:2].mean(),
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
        alpha=0.5,
    )
plt.xlim(-2, 10)
plt.ylim(-2, 10)
plt.title(f"Estimated number of clusters: {num_clusters}")
plt.show()


# extract outliers from the data 
outlier_indices = np.where(labels == -1)[0]
outliers = sounds2.iloc[outlier_indices]
# only 149 rows are considered outliers from above clustering

# filter out the outliers from the dataset 
sounds3 = sounds2[labels != -1]

#=========================== LOAD DATA =========================

# set seed
np.random.seed(42)

# load dataset and saperate X,y
X = sounds2.drop('Activity', axis=1)
y = sounds2['Activity']

# X,y shapes
print(f'X schema {X.shape}')
print(f'y schema {y.shape}')

# X,y head
print(X.head(3))
print(y.head(3))

# check features types
print(f'{X.dtypes}')
print(f'{y.dtypes}')

#============================== Modeling ==========================

# split dataset into training, testing sets 60-40
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the hyperparameters search space
hyperparameters_space = {
    'hidden_layer_sizes': Integer(1, 200),
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
# OrderedDict([('activation', 'tanh'), ('alpha', 1.505737910157384e-05), ('hidden_layer_sizes', 72), ('learning_rate_init', 0.007656584299243426), ('solver', 'adam')])
# OrderedDict([('activation', 'relu'), ('alpha', 2.132241999494801e-05), ('hidden_layer_sizes', 10), ('learning_rate_init', 0.0006871268023692144), ('solver', 'lbfgs')])
### OrderedDict([('activation', 'relu'), ('alpha', 0.001), ('hidden_layer_sizes', 19), ('learning_rate_init', 0.1), ('solver', 'lbfgs')])
print(search.best_score_)
# 0.9781553398058251 without PCA
### 0.9992716236051425 with Anne PCA
# 0.8519185566605086 with my PCA

# ============================== Evaluation =========================
# accurcy
best_model = search.best_estimator_
accuracy = cross_val_score(best_model, X_test, y_test, cv=5).mean()
print("Accuracy:", accuracy)
# with all features
# Accuracy: 0.9781553398058251
# My feature reduction
# Accuracy: 0.8274271844660195
# Anne feature reduction
# Accuracy: 0.9737864077669902

# confusion matrix
# predict classes of the test set
y_pred = best_model.predict(X_test) 

# Calculate confusion matrix of test
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




# ============================= MODELING ================================
# split dataset into training, testing sets 60-40

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
# My PCA Best score: 0.8297458171193479
# Anne PCA Best score: 0.9683211163574844
print(f'Best parameters:{random_search.best_params_}')
# Best parameters:{'random_state': 8, 'prototypes_per_class': 7, 'beta': 75}
# My PCA Best parameters:{'random_state': 8, 'prototypes_per_class': 7, 'beta': 75}
# Anne PCA Best parameters:{'random_state': 8, 'prototypes_per_class': 1, 'beta': 75}

#============================= Evaluation ===========================

# accurcy
best_model = random_search.best_estimator_
accuracy = cross_val_score(best_model, X_test, y_test, cv=5).mean()
print("Accuracy:", accuracy)
# Accuracy: 0.9769417475728155
# with feature reduction Accuracy: 0.8208737864077669
# Anne PCA Accuracy: 0.9655339805825243

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
som_width = 10
som_height = 10
som = MiniSom(som_width, som_height, X_train, sigma=1.0, learning_rate=0.5)
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


# Train the model
model = OneClassSVM(nu=0.1)
# nu hyperparameter controls proportion of data that is considered anomalous
#0.1 means that 10% of the data is considered anomalous.

model.fit(X_train)

# Predict anomalies on the testing set
predictions = model.predict(X_test)

# Convert the labels to binary (1: normal, -1: anomalous)
y_test_binary = np.where(y_test == 'normal', 1, -1)
predictions_binary = np.where(predictions == 1, 1, -1)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test_binary, predictions_binary).ravel()

# Print the confusion matrix
print('Confusion matrix:')
print('True positives:', tp)
print('False positives:', fp)
print('True negatives:', tn)
print('False negatives:', fn)

# Compute accuracy
accuracy = (tp + tn) / (tp + fp + tn + fn)
print('Accuracy:', accuracy)


# Get the decision function output for test set
y_score = model.decision_function(X_test)
# Compute the ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(predictions_binary, y_score)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for One-Class SVM')
plt.legend(loc="lower right")
plt.show()


'''it seems like the novelty detection model is not able 
to identify any anomalies in the test data, resulting in 
a high number of false positives and a low accuracy. 
This may be because there were no anomalies in the 
test set and all subjects are producing the same result, which means 
they are healthy, with no injury, no obesity!'''