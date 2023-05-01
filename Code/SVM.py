# %%
# SVM visualization
df=df_new.iloc[:10298,[0,1,20]]

ax = plt.axes(projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['Activity'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Activity')

#%%



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
cf_matrix = confusion_matrix(y_test, prediction)
sns.heatmap(cf_matrix, annot=True)

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
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

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
