# SALES PREDICTION WITH PYTHON

import numpy as np
import pandas as pd
# Load the Sales Prediction dataset
data = pd.read_csv(r"Sales_prediction.csv")
tv_ad_spend = np.array(data['TV'])
print("TV advertising spend:", tv_ad_spend)
radio_ad_spend = np.array(data['Radio'])
print("Radio advertising spend:", radio_ad_spend)
newspaper_ad_spend = np.array(data['Newspaper'])
print("Newspaper advertising spend:", newspaper_ad_spend)
sales = np.array(data['Sales'])
print("Sales:", sales)
print('\n\n')


# Perform various NumPy aggregations on the Sales column
sales = np.array(data['Sales'])
print("Min sales: ", sales.min())
print("Sum of sales: ", np.sum(sales))
print("Index of min sales: ", np.argmin(sales))
print("Index of max sales: ", np.argmax(sales))
print("Variance of sales: ", np.var(sales))
print("Median of sales: ", np.median(sales))
print("Mean of sales: ", sales.mean())
print("Max sales: ", sales.max())
print("75th percentile of sales: ", np.percentile(sales, 75))
print('\n\n')


# Relational Operators
print("Relational Operations:")
print("TV ad spend < 100: ", tv_ad_spend < 100)
print("Radio ad spend <= 50: ", radio_ad_spend <= 50)
print("Newspaper ad spend > 30: ", newspaper_ad_spend > 30)
print("Sales >= 200: ", sales >= 200)
print("Sales == 150: ", sales == 150)
print("TV ad spend != 75: ", tv_ad_spend != 75)
print('\n\n')


# Masking
print("Adding 10 to TV ad spend:", tv_ad_spend + 10)
print("Multiplying radio ad spend by 2:", radio_ad_spend * 2)
print("Selecting sales values greater than 250:", sales[sales > 250])
print('\n\n')


# Select 5 random rows using Fancy Indexing
random_indices = np.random.choice(data.index, 5, replace=False)  # Randomly select 5 indices
random_sales_data = data.loc[random_indices]  # Using indices to select rows
print("Randomly selected sales data:\n", random_sales_data)
print('\n\n')


import matplotlib.pyplot as plt
# Group the data by TV advertising spend ranges
bins = [0, 50, 100, 150, np.inf]
labels = ['Below 50', '50-100', '100-150', 'Above 150']
data['TV_Range'] = pd.cut(data['TV'], bins=bins, labels=labels, right=False)
# Calculate the average sales for each TV ad spend range
avg_sales = data.groupby('TV_Range', observed=False)['Sales'].mean()  # Set observed=False
# Create a bar graph
avg_sales.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Average Sales by TV Advertising Spend Range')
plt.xlabel('TV Advertising Spend Range')
plt.ylabel('Average Sales')
# Show the plot
plt.show()
print('\n\n')


# Create a vector of length 20 with values between the min and max of 'Sales'
sales = np.array(data['Sales'])
a = np.linspace(sales.min(), sales.max(), num=20)
# Print the vector
print("Vector of 20 evenly spaced values between min and max of Sales:")
print(a)
print('\n\n')


# Extract the 'Sales' column as a NumPy array
sales = np.array(data['Sales'])
# Normalize the sales data between 0 and 1
sales_normalized = (sales - sales.min()) / (sales.max() - sales.min())
# Create a vector of length 20 with values between the min and max of normalized 'Sales'
a = np.linspace(sales_normalized.min(), sales_normalized.max(), num=20)
# Print the original range and the normalized vector
print("Original Sales range: [", sales.min(), ",", sales.max(), "]")
print("Vector of 20 evenly spaced normalized values between 0 and 1:")
print(a)
print('\n\n')


# Select 'TV' and 'Sales'
data_tv = data[['TV']].set_index(data.index)
data_sales = data[['Sales']].set_index(data.index)
# Print the dataframes
print("TV DataFrame:")
print(data_tv, end="\n\n")
print("Sales DataFrame:")
print(data_sales, end="\n\n")
# Perform the join operation
joined_data = data_tv.join(data_sales, lsuffix="_TV", rsuffix="_Sales")
print("Joined DataFrame:")
print(joined_data)
print('\n\n')


# Create a pivot table
pivot_table = data.pivot_table(values='Sales', index='TV', aggfunc='mean')
# Print the pivot table
print("Pivot table - Average Sales by TV Advertising:")
print(pivot_table)
print('\n\n')


# Swap rows in reverse order
reversed_rows = data.iloc[::-1]
# Swap columns in reverse order
reversed_columns = data.iloc[:, ::-1]
# Swap both rows and columns in reverse order
reversed_both = data.iloc[::-1, ::-1]
print("Rows swapped in reverse order:")
print(reversed_rows)
print("\nColumns swapped in reverse order:")
print(reversed_columns)
print("\nBoth rows and columns swapped in reverse order:")
print(reversed_both)
print('\n\n')


# Convert 'TV' column values to string
data['TV_str'] = data['TV'].astype(str)
# String operations on the 'TV_str' column
# Converts the string to lowercase.
print("Lowercase:\n", data['TV_str'].str.lower())
# Finds the length of each string.
print("\nLength:\n", data['TV_str'].str.len())
# Checks if the string starts with "2"
print("\nStarts with '2':\n", data['TV_str'].str.startswith('2'))
print('\n\n')


# Convert 'TV' column values to string
data['TV_str'] = data['TV'].astype(str)
# Splits the string into a list
print("\nSplit:\n", data['TV_str'].str.split())
# Extracts a regex pattern from the string (extracts the first letter or digits)
print("\nExtract first digits or letters:\n", data['TV_str'].str.extract('([A-Za-z]+)', expand=False))
# Finds all instances of a regex pattern in the string.
# In this case, it checks patterns that don't start or end with vowels.
print("\nFind patterns that don't start/end with vowels:\n", data['TV_str'].str.findall(r'^[^AEIOU].*[^aeiou]$'))
# Gets the first 3 characters
print("\nFirst 3 characters:\n", data['TV_str'].str[0:3])
# Gets the last element after splitting the string
print("\nLast split element:\n", data['TV_str'].str.split().str.get(-1))
print('\n\n')


# Convert the Sales column into a NumPy array
sales = np.array(data['Sales'])
# Calculate mean, median, standard deviation, and variance
mean_sales = np.mean(sales)
median_sales = np.median(sales)
std_dev_sales = np.std(sales)
variance_sales = np.var(sales)
print("Mean of Sales:", mean_sales)
print("Median of Sales:", median_sales)
print("Standard deviation of Sales:", std_dev_sales)
print("Variance of Sales:", variance_sales)
print('\n\n')


# Select(TV, Radio, Newspaper, and Sales)
columns = ['TV', 'Radio', 'Newspaper', 'Sales']
# Convert the selected columns into a NumPy array
data_array = np.array(data[columns])
# Calculate the mean, standard deviation, and variance for each column
mean_values = np.mean(data_array, axis=0)
median_values = np.median(data_array, axis=0)
std_dev_values = np.std(data_array, axis=0)
variance_values = np.var(data_array, axis=0)
# Display the results
for i, column in enumerate(columns):
    print(f"Column: {column}")
    print(f"Mean: {mean_values[i]}")
    print(f"Median: {median_values[i]}")
    print(f"Standard Deviation: {std_dev_values[i]}")
    print(f"Variance: {variance_values[i]}")
    print('-' * 50)
print('\n\n')


# Sort the dataset by 'Sales' column
sorted_data = data.sort_values(by='Sales')
# Print the sorted dataset by 'Sales'
print("Sorted Dataset by Sales:")
print(sorted_data[['TV', 'Radio', 'Newspaper', 'Sales']])
print('\n\n')


# Sort the dataset by multiple columns: 'TV', 'Radio', 'Newspaper', and 'Sales'
sorted_data = data.sort_values(by=['TV', 'Radio', 'Newspaper', 'Sales'])
# Print the sorted dataset
print("Sorted Dataset by TV, Radio, Newspaper, and Sales:")
print(sorted_data[['TV', 'Radio', 'Newspaper', 'Sales']])
print('\n\n')


# Handling Missing Data
print("\nMissing Values Before Handling:")
print(data.isnull().sum())
# Select only numeric columns to fill missing values
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
# Fill missing values for numeric columns with their mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
# For categorical columns, you can fill missing values with the mode (most frequent value)
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
print("\nMissing Values After Handling:")
print(data.isnull().sum())
print('\n\n')


# Data Filtering
filtered_data = data[data['Sales'] > 1000]  # Replace 'Sales' with the actual column name if needed
print("\nFiltered Data (Sales > 1000):")
print(filtered_data.head())
print('\n\n')


# Data Transformation
import pandas as pd
from sklearn.preprocessing import StandardScaler
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
# Ensure there are numeric columns before proceeding
if len(numeric_columns) > 0:
    print("Numeric columns detected:", numeric_columns)
    # Standardize (Scale) the numeric columns using StandardScaler
    scaler = StandardScaler()
    # Make sure to handle cases where some non-numeric values might be present
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    # Perform scaling
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print("\nData After Transformation (Scaling):")
    print(data.head())
else:
    print("No numeric columns found in the dataset.")
print('\n\n')
