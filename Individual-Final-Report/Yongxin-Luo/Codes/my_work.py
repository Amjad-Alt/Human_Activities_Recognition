#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import pcolor, colorbar, plot
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score



#%%
sounds = pd.read_csv("D:/GraSem2/MachineLearning/Github/Final-Project_Grpup11/Code/data/sounds.csv")

#%%
X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]
#%%
# Scaling data
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

#%%
# Feature Reduction
pca = PCA(n_components = 20)
pca.fit(X_scaled)
X_pca = pca.fit_transform(X_scaled)

plt.bar(range(1,len(pca.explained_variance_ )+1),pca.explained_variance_ )
plt.ylabel('Explained variance')
plt.xlabel('Components')
plt.plot(range(1,len(pca.explained_variance_ )+1),
         np.cumsum(pca.explained_variance_),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='upper left')

#%%
df_new = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20'])
df_new['label'] = y
df_new.head()

#%%
# Creating a instance of label Encoder.
le = LabelEncoder()
# Using .fit_transform function to fit label
# encoder and return encoded label
label = le.fit_transform(df_new['label'])
# printing label
label

#%%
# removing the column 'Purchased' from df
# as it is of no use now.
df_new.drop("label", axis=1, inplace=True)
# Appending the array to our dataFrame
# with column name 'Purchased'
df_new["Activity"] = label
# printing Dataframe
df_new

#%%
X = df_new.iloc[:,:-1]
y = df_new.iloc[:,-1:]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%%
# PCA - Scatter Plot
fig = px.scatter(X_pca, x=0, y=1, color=sounds['Activity'])
fig.show()

#%%
# Making Correlation graphs using plt.matshow()
# X_train = pd.DataFrame(X_train)
# ax = sns.heatmap(pca.components_,
#                  cmap='YlGnBu',
#                  yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
#                  xticklabels=list(X_pca.columns),
#                  cbar_kws={"orientation": "horizontal"})
# ax.set_aspect("equal")
# corr = plt.matshow(df_new.corr())
# plt.colorbar(corr)
# sns.heatmap(df_new)
#%%
# Making Correlation Graph among PC1~20 and Activity using plt.imshow()
ax = plt.subplot()
im = ax.imshow(df_new.corr())
plt.xlabel("PC1~20 and Activity")
plt.ylabel("PC1~20 and Activity")
plt.title("Correlation among PC1~20 and Activity")
# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

#%%
# First two of components 
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('The first two component data')
plt.xlabel('C1')
plt.ylabel('C2')
plt.show()

# %%
# The first 10 components' explained_variance_ratio
pca.explained_variance_ratio_ *100

# %%
# SVM visualization
df=df_new.iloc[:10298,[0,1,20]]

ax = plt.axes(projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['Activity'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Activity')

# %%
# SVM
svc_model = SVC(C=.1, kernel='linear', gamma=1)
svc_model.fit(X_train, y_train)
 
prediction = svc_model.predict(X_test)
# check the accuracy on the training set
# print(svc_model.score(y_test, prediction))

print(accuracy_score(y_test, prediction))
print(precision_score(y_test, prediction, average='macro'))
print(recall_score(y_test, prediction, average='macro'))
print(f1_score(y_test, prediction, average='macro'))
scores = cross_val_score(svc_model, y_test, prediction, cv=3,scoring='accuracy')
print("Score of Cross Validation:" + str(scores.mean()))
cf_matrix = confusion_matrix(y_test, prediction)
sns.heatmap(cf_matrix, annot=True)

#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
#                                  index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')

#%%
support_vector_indices = svc_model.support_
print(support_vector_indices)

support_vectors_per_class = svc_model.n_support_
print(support_vectors_per_class)

# Get support vectors themselves
support_vectors = svc_model.support_vectors_

# Visualize support vectors
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

#%%
# Linear-SVM Model
poly_svc=SVC(kernel='poly', C=10) 
poly_svc.fit(X_train,y_train)
# make predictions on test set
y_pred_poly=poly_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=10 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_poly)))
print(f"Precision score: {precision_score(y_test, y_pred_poly, average='macro')}")
print(f"Recall rate: {recall_score(y_test, y_pred_poly, average='macro')}")
print(f"f1-score: {f1_score(y_test, y_pred_poly, average='macro')}")
cf_matrix = confusion_matrix(y_test, prediction)
sns.heatmap(cf_matrix, annot=True)

#%%
# Sigmoid SVM model
sigmoid_svc=SVC(kernel='sigmoid', C=1.0) 
# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)
# make predictions on test set
y_pred_sig = sigmoid_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_sig)))
print(f"Precision score: {precision_score(y_test, y_pred_sig, average='macro')}")
print(f"Recall rate: {recall_score(y_test, y_pred_sig, average='macro')}")
print(f"f1-score: {f1_score(y_test, y_pred_sig, average='macro')}")

#%%
# Checking over-fitting or under-fitting
print('Training set score: {:.4f}'.format(svc_model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(svc_model.score(X_test, y_test)))

print('Training set score: {:.4f}'.format(poly_svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(poly_svc.score(X_test, y_test)))

print('Training set score: {:.4f}'.format(sigmoid_svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(sigmoid_svc.score(X_test, y_test)))


#%%
# hyperparamater optimization - SVC grid search
rf_params = {
    'C': [1,10, 100],
    "kernel":['linear','poly','rbf','sigmoid']
}
svc_grid = SVC(gamma='scale')
grid = GridSearchCV(svc_grid, rf_params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Accuracy:"+ str(grid.best_score_))

#%%
# Bayesian Optimization with Gaussian Process
from skopt import Optimizer
from skopt import BayesSearchCV 
from skopt.space import Real, Categorical, Integer
rf_params = {
    'C': Real(0.01,50),
    "kernel":['linear','poly','rbf','sigmoid']
}
clf = SVC(gamma='scale')
Bayes = BayesSearchCV(clf, rf_params,cv=3,n_iter=20, n_jobs=-1,scoring='accuracy')
Bayes.fit(X_train, y_train)
print(Bayes.best_params_)
bclf = Bayes.best_estimator_
print("Accuracy:"+ str(Bayes.best_score_))

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def create_model(): 
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='softmax')
    ])
    return model

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping()
custom_early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=42, 
    min_delta=0.001, 
    mode='max'
)

model = create_model()
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    validation_split=0.25, 
    batch_size=20, 
    verbose=2,
    callbacks=[custom_early_stopping]
)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
