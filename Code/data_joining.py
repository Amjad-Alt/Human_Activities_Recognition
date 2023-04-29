
################# Do not run these codes it is one time thing ########################


#======================== Join train/test datasets =================================

# Set the working directory to the directory containing the datasets
os.chdir("C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/data")

# Create an empty list to store the dataframes
df_list = []

# Loop through the files in the directory
for file in os.listdir():
    # Check if the file is a CSV file
    if file.endswith(".csv"):
        # Read the CSV file into a pandas dataframe
       df = pd.read_csv(file)
        # Append the dataframe to the list
       df_list.append(df)

# Concatenate the dataframes in the list
concatenated_df = pd.concat(df_list, axis=0)

# save it as one file
concatenated_df.to_csv("sounds.csv", index=False)

# ================== Join raw data of single user ============================

# Define the directory where the files are stored
data_dir = "C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/data/HAPT_Data_Set/RawData"
save_dir = "C:/Users/amjad/OneDrive/المستندات/GWU_Cources/Spring2023/Machine_Learning/MLproject/Code/data"
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
