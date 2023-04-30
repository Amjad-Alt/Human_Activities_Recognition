

#=========================== LOAD DATA =========================

# set seed
np.random.seed(42)

# train and split from the original data without deleteing outliers from the clusters

sounds2.head()
X_2 = sounds2.drop('Activity', axis=1)
y_2 = sounds2['Activity']

# X,y shapes
print(f'X schema {X_2.shape}')
print(f'y schema {y_2.shape}')

# X,y head
print(X_2.head(3))
print(y_2.head(3))

# check features types
print(f'{X_2.dtypes}')
print(f'{y_2.dtypes}')

# split dataset into training, testing sets 60-40
X_onetrain, X_valtest, y_onetrain, y_valtest = train_test_split(X_2, y_2, test_size=0.4, random_state=42)

# Train the model
model = OneClassSVM(nu=0.05)
# nu hyperparameter controls proportion of data that is considered anomalous
#0.1 means that 10% of the data is considered anomalous.

model.fit(X_onetrain)

# Predict anomalies on the testing set
predictions = model.predict(X_valtest)

# Convert the labels to binary (1: normal, -1: anomalous)
y_test_binary = np.where(y_valtest == 'normal', 1, -1)
predictions_binary = np.where(predictions == 1, 1, -1)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test_binary, predictions_binary).ravel()

# Print the confusion matrix
print('Confusion matrix:')
print('True positives:', tp)
print('False positives:', fp)
print('True negatives:', tn)
print('False negatives:', fn)

# Confusion matrix:
# True positives: 0
# False positives: 3685
# True negatives: 435
# False negatives: 0

# Compute accuracy
accuracy = (tp + tn) / (tp + fp + tn + fn)
print('Accuracy:', accuracy)
#Accuracy: 0.10558252427184465


# Get the decision function output for test set
y_score = model.decision_function(X_valtest)
# Compute the ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(predictions_binary, y_score)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for One-Class SVM')
plt.legend(loc="lower right")
plt.show()


'''it seems like the novelty detection model is not able 
to identify any anomalies in the test data, resulting in 
a high number of false positives and a low accuracy. 
This may be because there were no anomalies in the 
test set and all subjects are producing the same result, which means 
they are healthy, with no injury, no obesity!'''