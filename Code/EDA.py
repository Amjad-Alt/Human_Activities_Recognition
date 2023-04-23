# import packages
import pandas as pd
import numpy as np
import os

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# clustering
from collections import Counter
from sklearn.cluster import DBSCAN 



#----------------------LOAD DATA --------------------------

#=======================Grouping test and train =================================
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


# ================================== Simple EDA ====================================
# set seed
np.random.seed(42)

# load dataset and saperate X,y
sounds = pd.read_csv("sounds.csv")

#X = sounds.iloc[:,:-1]
#y = sounds.iloc[:,-1:]

# shuffle the data
# X, y = shuffle(X, y, random_state=42)

# Understand data such as NA valuse and do graphs and data type

# data shapes
print(f'whole dataset schema {sounds.shape}')
#print(f'whole dataset schema {X.shape}')
#print(f'whole dataset schema {y.shape}')

# check Na values
print(f'number of Na in the whole dataframe {sounds.isna().sum().sum()}')

# data head
print(sounds.head(3))
#print(X.head(3))
#print(y.head(3))

# check features types
print(f'{sounds.dtypes}')
#print(f'{X.dtypes}')
#print(f'{y.dtypes}')

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

# change y into numarical
print(f'Activities before numarically label them {sounds2.iloc[:,-1].unique()}')
sounds2['Activity'] = sounds2['Activity'].replace({'STANDING': 1, 'SITTING': 2, 'LAYING': 3, 'WALKING': 4, 'WALKING_DOWNSTAIRS':5, 'WALKING_UPSTAIRS':6})
print(f'Activities values after labeling {sounds2.iloc[:,-1].unique()}')


# =============================== plotting 3D ========================================

# it needs working on like mayby using the raw daatset
ax = plt.axes(projection='3d')
ax.plot_surface(sounds2['tBodyAcc-mean()-X'],
                sounds2['tBodyAcc-mean()-Y'],
                sounds2['tBodyAcc-mean()-Z'], 
                rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')


#=============================== Clustering data to detect outliers ==================================
# Fit DBSCAN to the data
clustering = DBSCAN(eps=5, min_samples=2).fit(sounds2)
labels = clustering.labels_


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
plt.xlim(-5, 35)
plt.ylim(-5, 35)
plt.title(f"Estimated number of clusters: {num_clusters}")
plt.show()

# extract outliers from the data 
outlier_indices = np.where(labels == -1)[0]
outliers = sounds2.iloc[outlier_indices]

# filter out the outliers from the dataset 
sounds3 = sounds2[labels != -1]