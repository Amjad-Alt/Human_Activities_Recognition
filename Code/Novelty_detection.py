
# Train the model
model = OneClassSVM(nu=0.1)
# nu hyperparameter controls proportion of data that is considered anomalous
#0.1 means that 10% of the data is considered anomalous.

model.fit(X_train)

# Predict anomalies on the testing set
predictions = model.predict(X_test)

# Convert the labels to binary (1: normal, -1: anomalous)
y_test_binary = np.where(y_test == 'normal', 1, -1)
predictions_binary = np.where(predictions == 1, 1, -1)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test_binary, predictions_binary).ravel()

# Print the confusion matrix
print('Confusion matrix:')
print('True positives:', tp)
print('False positives:', fp)
print('True negatives:', tn)
print('False negatives:', fn)

# Compute accuracy
accuracy = (tp + tn) / (tp + fp + tn + fn)
print('Accuracy:', accuracy)


# Get the decision function output for test set
y_score = model.decision_function(X_test)
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