import pandas as pd

# Read the CSV file into a DataFrame
euro = pd.read_csv("GOOG.US_D1.csv")

# Transpose the DataFrame
dfEuro = pd.DataFrame(euro).T

# Set display options to show more rows and columns
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# Print the transposed DataFrame
print(dfEuro)

# Print the first and last row of the transposed DataFrame
print("\n___HEAD___\n", dfEuro.head(1))
print("\n___TAIL___\n", dfEuro.tail(1))

# Compute the mean of the 'high' column after filling missing values with the mean
goal = euro['high']
goal_mean = goal.mean()
goal.fillna(goal_mean, inplace=True)
euro.isnull().sum()
print("\nMean of 'high' column after filling missing values:", goal_mean)




# Check for missing values
missing_values = euro.isnull().sum()
print("Missing values:\n", missing_values)

# Drop rows with any missing values
euro_cleaned = euro.dropna()

# Check for duplicates
duplicates = euro_cleaned.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)

# Drop duplicate rows
euro_cleaned = euro_cleaned.drop_duplicates()

# Compute the mean of the 'high' column after handling missing values
goal = euro_cleaned['high']
goal_mean = goal.mean()
print("\nMean of 'high' column:", goal_mean)

# Print cleaned DataFrame
print("\nCleaned DataFrame:\n", euro_cleaned.head())

# Save cleaned DataFrame to a new CSV file
euro_cleaned.to_csv("GOOG.US_D1_cleaned.csv", index=False)
