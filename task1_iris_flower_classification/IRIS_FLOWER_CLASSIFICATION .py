#IRIS FLOWER CLASSIFICATION

import numpy as np
import pandas as pd
# Load the Iris dataset
data = pd.read_csv(r"iris_flower.csv")
petal_length = np.array(data['petal_length'])
print("petal_length:", petal_length)
sepal_length = np.array(data['sepal_length'])
print("sepal_length:", sepal_length)
petal_width = np.array(data['petal_width'])
print("petal_width:", petal_width)
sepal_width = np.array(data['sepal_width'])
print("sepal_width:", sepal_width)
print('\n\n')

# Perform various NumPy aggregations
print("min petal length: ", petal_length.min())
print("sum: ", np.sum(petal_length))
print("min index: ", np.argmin(petal_length))
print("max index: ", np.argmax(petal_length))
print("variance: ", np.var(petal_length))
print("median: ", np.median(petal_length))
print("mean: ", petal_length.mean())
print("max petal length: ", petal_length.max())
print("75th percentile: ", np.percentile(petal_length, 75))
print('\n\n')

# Relational Operators
sepal_length = np.array(data['sepal_length'])
petal_length = np.array(data['petal_length'])
print("Relational Operations:")
print("sepal_length < 3: ", sepal_length < 3)
print("sepal_length <= 6: ", sepal_length <= 6)
print("sepal_length > 4: ", sepal_length > 4)
print("sepal_length >= 6: ", sepal_length >= 6)
print("sepal_length == 5: ", sepal_length == 5)
print("sepal_length != 7: ", sepal_length != 7)
print('\n\n')


# Masking
print("Adding 3 to sepal_length:", sepal_length + 3)
print("Multiplying sepal_length by 3:", (sepal_length * 3))
print("Selecting petal_length values greater than 1.5:", petal_length[petal_length > 1.5])
print('\n\n')


# Select 5 random flowers using Fancy Indexing
random_indices = np.random.choice(data.index, 5, replace=False)  # Randomly select 5 indices
random_flowers = data.loc[random_indices]  # indices to select rows
print("Randomly selected flowers:", random_flowers)
print('\n\n')


import matplotlib.pyplot as plt
# Load the Iris dataset
data = pd.read_csv(r"iris_flower.csv")
# Group the data by species and calculate the mean of sepal_length for each species
avg_sepal_length = data.groupby('species')['sepal_length'].mean()
# Create a bar graph
avg_sepal_length.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
# Show the plot
plt.show()
print('\n\n')


# Create a vector of length 20 with values  between the minimum and maximum of 'sepal_length'
a = np.linspace(sepal_length.min(), sepal_length.max(), num=20)
print(a)


# join--> combining dataset
import pandas as pd
data = pd.read_csv(r"iris_flower.csv")
# Select:'sepal_length' and 'petal_length'
data_sepal = data[['sepal_length']].set_index(data.index)
data_petal = data[['petal_length']].set_index(data.index)
# Print the dataframes
print("Sepal DataFrame:")
print(data_sepal, end="\n\n")
print("Petal DataFrame:")
print(data_petal, end="\n\n")
# Perform the join operation
joined_data = data_sepal.join(data_petal, lsuffix="_sepal", rsuffix="_petal")
print("Joined DataFrame:", joined_data)
print('\n\n')


# Join--> pandas to cobining datasets
data_sepal = data[['sepal_length']].set_index(data.index)
data_petal = data[['petal_length']].set_index(data.index)
print(data_sepal)
print(data_petal)
data_sepal.join(data_petal, lsuffix="_sepal", rsuffix="_petal")
print('\n\n')


# pandas using pivot table for iris datset
pivot_table = data.pivot_table(values='sepal_length', index='sepal_width', aggfunc='mean')
print("Pivot table-Average Sepal Length by Sepal Width:")
print(pivot_table)
print('\n\n')


#swaping rows and columns in reverse order
import pandas as pd
import numpy as np
# Load your dataset (adjust the file path as needed)
data = pd.read_csv(r"iris_flower.csv")
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


# pandas using vetorized string operations
import pandas as pd
data['sepal_length_str'] = data['sepal_length'].astype(str)
# String operations
#Converts the string to lowercase.
print("Lowercase:\n", data['sepal_length_str'].str.lower())
# Finds the length of each string.
print("\nLength:\n", data['sepal_length_str'].str.len())
#Checks if the string starts with "5
print("\nStarts with '5':\n", data['sepal_length_str'].str.startswith('5'))
#Splits the string into a list.
print("\nSplit:\n", data['sepal_length_str'].str.split())
# Extracts a regex pattern from the string.
print("\nExtract letters:\n", data['sepal_length_str'].str.extract('([A-Za-z]+)', expand=False))
#Finds all instances of a regex pattern in the string.
print("\nFind patterns that don't start/end with vowels:\n", data['sepal_length_str'].str.findall(r'^[^AEIOU].*[^aeiou]$'))
#Gets the first 3 characters.
print("\nFirst 3 characters:\n", data['sepal_length_str'].str[0:3])
#Gets the last element after splitting the string.
print("\nLast split element:\n", data['sepal_length_str'].str.split().str.get(-1))
print('\n\n')


# Calculaing the mean, standard deviation and  variance
import numpy as np
import pandas as pd
data = pd.read_csv(r"iris_flower.csv")
# Convert the petal_width column into a NumPy array and reshape it to be 2D (for axis=1)
petal_width = np.array(data[['petal_width']])
# Calculate mean, median, standard deviation, and variance along the second axis (rows)
mean = np.mean(petal_width, axis=1)
median = np.median(petal_width, axis=1)
std_dev = np.std(petal_width, axis=1)
variance = np.var(petal_width, axis=1)
print("Mean along the second axis (petal_width):", mean)
print("Median along the second axis (petal_width):", median)
print("Standard deviation along the second axis (petal_width):", std_dev)
print("Variance along the second axis (petal_width):", variance)
print('\n\n')


# Select relevant columns for analysis (sepal_length, sepal_width, petal_length, and petal_width)
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# Convert the selected columns into a NumPy array
data_array = np.array(data[columns])
# Calculate the mean, median, standard deviation, and variance for each column
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


# Sorting
import pandas as pd
data = pd.read_csv(r"iris_flower.csv")
# Sort the dataset by 'petal_width'
sorted_data = data.sort_values(by='petal_width')
# Print the sorted dataset
print("Sorted Dataset by Petal Width:")
print(sorted_data[['petal_width']])
print('\n\n')


# Sort the dataset by multiple columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
sorted_data = data.sort_values(by=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
# Print the sorted dataset
print("Sorted Dataset by Sepal Length, Sepal Width, Petal Length, and Petal Width:")
print(sorted_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
print('\n\n')


# Load the dataset
from sklearn.datasets import load_iris
data = load_iris(as_frame=True).frame
# Check for missing values
print("Before Handling Missing Data:")
print(data.isnull().sum())
# Fill missing values only in numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
# data.dropna(inplace=True)
print("\nAfter Handling Missing Data:")
print(data.isnull().sum())
print('\n\n')


# Filter rows where sepal length is greater than 5
filtered_data = data[data['sepal length (cm)'] > 5]
print("\nFiltered Data (Sepal length > 5):")
print(filtered_data.head())
print('\n\n')

# Create a new column for Sepal Area (Sepal Length * Sepal Width)
data['sepal area (cm^2)'] = data['sepal length (cm)'] * data['sepal width (cm)']
print("\nData After Transformation (Added Sepal Area):")
print(data.head())
print('\n\n')

# Concatenate two copies of the iris dataset horizontally
data_concat = pd.concat([data, data], axis=1)
print("\nConcatenated Data:")
print(data_concat.head())
print('\n\n')

# Check the column names to ensure the target column is named correctly
print("Column Names:", data.columns)
# Check the first few rows of the dataset to ensure it is loaded correctly
print("\nFirst few rows of the dataset:",data.head())
target_column = 'species'
# Convert the target column (species) to categorical type
if target_column in data.columns:
    data[target_column] = data[target_column].astype('category')
    print(f"\n'{target_column}' column converted to categorical type.")
else:
    print(f"\n'{target_column}' column not found in the dataset.")
print('\n\n')

# Display the data types after conversion
print("\nData Types After Conversion:")
print(data.dtypes)
print('\n\n')
