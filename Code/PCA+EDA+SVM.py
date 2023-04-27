#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC

#%%
sounds = pd.read_csv("sounds.csv")
plt.matshow(sounds.corr())

#%%
X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]
#%%
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps

#%%
pca = PCA(n_components = 10)
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
df_new = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
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
X_train, X_test, y_train, y_test = train_test_split(df_new, y, test_size = 0.2, random_state = 0)

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
corr = plt.matshow(df_new.corr())
plt.colorbar(corr)
# sns.heatmap(df_new)
#%%
# Making Correlation Graph among PC1~10 and Activity using plt.imshow()
ax = plt.subplot()
im = ax.imshow(df_new.corr())
plt.xlabel("PC1~10 and Activity")
plt.ylabel("PC1~10 and Activity")
plt.title("Correlation among PC1~10 and Activity")
# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()

#%%
# First two of components 
plt.scatter(X_pca[:, 0], X_pca[:, 1])
# %%
# The first 10 components' explained_variance_ratio
pca.explained_variance_ratio_ *100


# %%
# SVM
svc_model = SVC(C=.1, kernel='linear', gamma=1)
svc_model.fit(X_train, y_train)
 
prediction = svc_model .predict(X_test)
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))
# %%
# SVM visualization
df=df_new.iloc[:10298,[0,1,10]]

ax = plt.axes(projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['Activity'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Activity')
# %%
