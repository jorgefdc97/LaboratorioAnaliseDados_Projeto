
import pandas as pd


# Load the CSV file
file_path = 'GOOG.US_MN1.csv'
data = pd.read_csv(file_path)

# Step 1: Remove duplicate rows
data_cleaned = data.drop_duplicates()

# Step 1: Drop rows with more than 50% missing values
threshold = len(data.columns) * 0.5
data_cleaned = data.dropna(thresh=threshold)

# Step 2: Handle missing values
# Checking the percentage of missing values in each column
missing_values = data_cleaned.isnull().mean() * 100

# Step 3: Drop columns with a high percentage of missing values (e.g., more than 50%)
columns_to_drop = missing_values[missing_values > 50].index
data_cleaned = data_cleaned.drop(columns=columns_to_drop)

# Step 2: Fill missing values with the mean of each numeric column divided by the number of rows
numeric_columns = data_cleaned.select_dtypes(include='number').columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].mean())

# Optional: Save the cleaned data to a new CSV file
data_cleaned.to_csv('GOOG.US_MN1_cleaned.csv', index=False)

'''
file_path = 'GOOG.US_W1.csv'
data = pd.read_csv(file_path)

# Step 1: Remove duplicate rows
data_cleaned = data.drop_duplicates()

# Step 1: Drop rows with more than 50% missing values
threshold = len(data.columns) * 0.5
data_cleaned = data.dropna(thresh=threshold)

# Step 2: Handle missing values
# Checking the percentage of missing values in each column
missing_values = data_cleaned.isnull().mean() * 100

# Step 3: Drop columns with a high percentage of missing values (e.g., more than 50%)
columns_to_drop = missing_values[missing_values > 50].index
data_cleaned = data_cleaned.drop(columns=columns_to_drop)

# Step 2: Fill missing values with the mean of each numeric column divided by the number of rows
numeric_columns = data_cleaned.select_dtypes(include='number').columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].mean())

# Optional: Save the cleaned data to a new CSV file
data_cleaned.to_csv('GOOG.US_W1_cleaned.csv', index=False)
'''

