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

