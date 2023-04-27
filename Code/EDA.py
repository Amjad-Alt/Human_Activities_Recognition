

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
# ============================== Carrolation Matrix==========================

# carrolation best features
corr_matrix = sounds2.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Get the top 10 features with the highest correlation
top10 = corr_matrix.nlargest(10, 'Activity')['Activity'].index
# Find index of feature columns with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
# Drop highly correlated features
top10 = sounds2.drop(sounds2[to_drop], axis=1, inplace=True)

# most carrolated columns to the target `Activity`
# Index(['Activity', 'fBodyAcc-entropy()-X', 'tBodyAcc-sma()',
#       'fBodyAccJerk-entropy()-X', 'tBodyAccMag-mean()', 'tBodyAccMag-sma()',
#       'tGravityAccMag-mean()', 'tGravityAccMag-sma()',
#       'tBodyAccJerk-entropy()-X', 'tBodyGyro-sma()'],
#      dtype='object')


# Plot heatmap of the correlation matrix
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix of Sound Recognition', fontsize=16)
plt.show()


#=============================== Detect outliers ==================================

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