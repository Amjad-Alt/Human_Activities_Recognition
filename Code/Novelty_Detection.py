# important pk
import pandas as pd

# model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# validation
from sklearn.metrics import confusion_matrix, classification_report

# plot
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================== Join raw data of single use ============================

# Define the directory where the files are stored
data_dir = "C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/HAPT_Data_Set/RawData"
save_dir = "C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code"
# Loop users
for user in range(1, 31):
    
    # Define the filenames for this user's accelerometer and gyroscope data
    acc_filenames = [f"{data_dir}/acc_exp02_user01.txt", f"{data_dir}/acc_exp02_user01.txt"]
    gyro_filenames = [f"{data_dir}/gyro_exp01_user01.txt", f"{data_dir}/gyro_exp02_user01.txt"]
    
    # Read in the accelerometer and gyroscope data for this user
    acc_data = pd.concat([pd.read_csv(f, header=None, sep=" ") for f in acc_filenames])
    gyro_data = pd.concat([pd.read_csv(f, header=None, sep=" ") for f in gyro_filenames])
    
    # Join the accelerometer and gyroscope data horizontally
    data_h = pd.concat([acc_data, gyro_data], axis=1)

    # Add column names
    data_h.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Save the joined data to a file for this user
    data_h.to_csv(f"{save_dir}/user01_data.csv", index=False)

# =================================== One-class SVM User1==================================

# Load the dataset
data = pd.read_csv('user01_data.csv')

data.shape

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Scale the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Train the model
model = OneClassSVM(nu=0.1)
# nu hyperparameter controls proportion of data that is considered anomalous
#0.1 means that 10% of the data is considered anomalous.

model.fit(train_data_scaled)

# Predict anomalies
predictions = model.predict(test_data_scaled)
anomalies = test_data[predictions < 0]


#========================= plot model result ===========================
# need to fix the y axies
# Plot the predictions and anomalies
palette = sns.color_palette("husl", 3)

# Create figure and axes objects
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Plot test data
ax1.plot(test_data.index, test_data, color=palette[0])
ax1.set_title('Test Data')
ax1.set_ylim(-8, 5)

# Plot predicted normal data
normal_data = test_data[predictions == 1]
ax2.plot(normal_data.index, normal_data, color=palette[1])
ax2.set_title('Predicted Normal Data')
ax2.set_ylim(-8, 5)

# Plot anomalies
ax3.plot(anomalies.index, anomalies, color=palette[2])
ax3.set_title('Anomalies')
ax3.set_ylim(-8, 5)

# Add figure title and legend
fig.suptitle('One-Class SVM Results User1', fontsize=14, fontweight='bold')
plt.show()

# ===================================== Evaluate =======================

# Evaluate the model
precision = len(anomalies) / len(test_data)
recall = len(anomalies) / len(data)
f1_score = 2 * precision * recall / (precision + recall)

# Print evaluation metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")



# =================================== One-class SVM User2==================================

# Load the dataset
data2 = pd.read_csv('user02_data.csv')

data2.shape

# Split the data into training and testing sets
train_data2 = data2.sample(frac=0.8, random_state=42)
test_data2 = data2.drop(train_data2.index)

# Scale the data
train_data_scaled2 = scaler.fit_transform(train_data2)
test_data_scaled2 = scaler.transform(test_data2)

model.fit(train_data_scaled2)

# Predict anomalies
predictions2 = model.predict(test_data_scaled2)
anomalies2 = test_data2[predictions2 < 0]


#========================= plot model result ===========================
# need to fix the y axies
# Plot the predictions and anomalies
palette = sns.color_palette("husl", 3)

# Create figure and axes objects
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Plot test data
ax1.plot(test_data2.index, test_data2, color=palette[0])
ax1.set_title('Test Data')
ax1.set_ylim(-7, 6)

# Plot predicted normal data
normal_data2 = test_data2[predictions2 == 1]
ax2.plot(normal_data2.index, normal_data2, color=palette[1])
ax2.set_title('Predicted Normal Data')
ax2.set_ylim(-7, 6)

# Plot anomalies
ax3.plot(anomalies2.index, anomalies2, color=palette[2])
ax3.set_title('Anomalies')
ax3.set_ylim(-7, 6)

# Add figure title and legend
fig.suptitle('One-Class SVM Results User2', fontsize=14, fontweight='bold')
plt.show()

# ===================================== Evaluate =======================

# Evaluate the model
precision2 = len(anomalies2) / len(test_data2)
recall2 = len(anomalies2) / len(data2)
f1_score2 = 2 * precision2 * recall2 / (precision2 + recall2)

# Print evaluation metrics
print(f"Precision: {precision2}")
print(f"Recall: {recall2}")
print(f"F1 Score: {f1_score2}")
