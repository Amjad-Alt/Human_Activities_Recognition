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
import plotly.express as px

# data prepocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

# clustering
from collections import Counter
from sklearn.cluster import DBSCAN 

# modeling 
from sklearn.neural_network import MLPClassifier # model chosen
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn_lvq import GlvqModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import OneClassSVM
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint

# model evaluation
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

