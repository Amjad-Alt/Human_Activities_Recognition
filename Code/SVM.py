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
# print(svc_model.score(y_test, prediction))
scores = cross_val_score(svc_model, y_test, prediction, cv=3,scoring='accuracy')
print("Score of Cross Validation:" + str(scores.mean()))
print(accuracy_score(y_test, prediction))
print(precision_score(y_test, prediction, average='macro'))
print(recall_score(y_test, prediction, average='macro'))
print(f1_score(y_test, prediction, average='macro'))
print(f1_score(y_test, prediction, average='weighted'))

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
print('Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))

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
