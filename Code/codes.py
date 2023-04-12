# Import packages
from sklearn.neural_network import MLPClassifier # model chosen
from sklearn.model_selection import train_test_split 
from bayes_opt import BayesianOptimization # bysian for alpha 
import pandas as pd
import numpy as np
import os

#----------------------LOAD DATA --------------------------

# Set the working directory to the directory containing the datasets
#os.chdir("C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code")

# Create an empty list to store the dataframes
#df_list = []

# Loop through the files in the directory
#for file in os.listdir():
    # Check if the file is a CSV file
    #if file.endswith(".csv"):
        # Read the CSV file into a pandas dataframe
       #df = pd.read_csv(file)
        # Append the dataframe to the list
        #df_list.append(df)

# Concatenate the dataframes in the list
# concatenated_df = pd.concat(df_list, axis=0)

# save it as one file
# concatenated_df.to_csv("sounds.csv", index=False)


# load dataset and saperate X,y
sounds = pd.read_csv("sounds.csv")
X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]

# Understand data such as NA valuse and do graphs and data type

# data shapes
print(f'whole dataset schema {sounds.shape}')
print(f'whole dataset schema {X.shape}')
print(f'whole dataset schema {y.shape}')

# check Na values
print(f'number of Na in the whole dataframe {sounds.isna().sum().sum()}')

# data head
print(X.head(3))
print(y.head(3))

# check features types
print(f'{X.dtypes}')
print(f'{y.dtypes}')

# change y into numarical
print(f'y values before numarically label them {y.iloc[:,0].unique()}')
y = y.replace({'STANDING': 1, 'SITTING': 2, 'LAYING': 3, 'WALKING': 4, 'WALKING_DOWNSTAIRS':5, 'WALKING_UPSTAIRS':6})
print(f'y values after labeling {y.iloc[:,0].unique()}')


#---------------------PREPARE DATA ------------------------

# split the dataset into training, validation, and testing sets
# 60-40
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.4, random_state=42)
# split test 50-50
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)


# define the objective function to optimize
def objective(alpha, hidden_layer_sizes, max_iter):
    # define the MLPClassifier model with the given hyperparameters
    mlp = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    
    # train the model on the training set
    mlp.fit(X_train, y_train)
    
    # evaluate the model on the testing set and return the negative accuracy
    # negative accuracy used instead of the regular accuracy
    # because Bayesian optimization aims to maximize objective function,
    # so we need to use the negative of the accuracy to turn it into a minimization problem.'''
    return -mlp.score(X_test, y_test)

# define the hyperparameter search space
pbounds = {
    'alpha': (0.0001, 1.0),#range of values that can be explored
    'hidden_layer_sizes': (10, 100),
    'max_iter': (100, 1000)
}

# create the BayesianOptimization object and run the optimization
# there is an issue with the bysian model
bo = BayesianOptimization(f=objective, pbounds=pbounds)
bo.maximize(n_iter=10)

# print the best hyperparameters and their corresponding score
print('Best hyperparameters:', bo.max)
bo.best_params_

