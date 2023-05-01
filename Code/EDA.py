

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

palette = sns.color_palette("Set2", 6)

# Create figure and axes objects
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 20))

# Plot 
ax1.plot(data.iloc[:,0], data.iloc[:,0], color=palette[0])
ax1.set_title('accelerometer X')
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
fig.suptitle('Subjcet1 Gyroscope|Accelerometer', fontsize=14, fontweight='bold')
plt.show()


data2 = pd.read_csv('data/user02_data.csv')

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 20))

# Plot 
ax1.plot(data2.iloc[:,0], data2.iloc[:,0], color=palette[0])
ax1.set_title('accelerometer X')
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
fig.suptitle('Subject2 Gyroscope|Accelerometer', fontsize=14, fontweight='bold')
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

# PCA
X = sounds.iloc[:,:-1]
y = sounds.iloc[:,-1:]
#%%
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps

#%%
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
corr = plt.matshow(df_new.corr())
plt.colorbar(corr)
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