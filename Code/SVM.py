# %%
# SVM visualization
df=df_new.iloc[:10298,[0,1,10]]

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
print(svc_model.score(y_test, prediction))
scores = cross_val_score(svc_model, y_test, prediction, cv=3,scoring='accuracy')
print("Score of Cross Validation:" + str(scores.mean()))
accuracy_score(y_test, prediction)
f1_score(y_test, prediction, average='macro')
f1_score(y_test, prediction, average='weighted')

#%%
# SVC grid search
rf_params = {
    'C': [1,10, 100],
    "kernel":['linear','poly','rbf','sigmoid']
}
svc2 = SVC(gamma='scale')
grid = GridSearchCV(svc2, rf_params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
print("Accuracy:"+ str(grid.best_score_))

#%%
# rf_params = {
#     'C': stats.uniform(0,50),
#     "kernel":['linear','poly','rbf','sigmoid']
# }
# n_iter_search=20
# clf = SVC(gamma='scale')
# Random = RandomizedSearchCV(clf, param_distributions=rf_params,n_iter=n_iter_search,cv=3,scoring='accuracy')
# Random.fit(X_train, y_train)
# print(Random.best_params_)
# print("Accuracy:"+ str(Random.best_score_))


# %%
# Voting Classifier
