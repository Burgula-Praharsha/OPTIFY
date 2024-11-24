import numpy as np
import pandas as pd

# Load the Car Price Prediction dataset
data = pd.read_csv(r"car_price_prediction_ML.csv")

# Extract features as NumPy arrays
car_names = np.array(data['Car_Name'])
print("Car Names:", car_names)

manufacturing_year = np.array(data['Year'])
print("Manufacturing Year:", manufacturing_year)

selling_price = np.array(data['Selling_Price'])
print("Selling Price:", selling_price)

present_price = np.array(data['Present_Price'])
print("Present Price:", present_price)

driven_kms = np.array(data['Driven_kms'])
print("Kilometers Driven:", driven_kms)

fuel_type = np.array(data['Fuel_Type'])
print("Fuel Type:", fuel_type)

selling_type = np.array(data['Selling_type'])
print("Selling Type:", selling_type)

transmission = np.array(data['Transmission'])
print("Transmission Type:", transmission)

owner = np.array(data['Owner'])
print("Owner Count:", owner)

# Perform NumPy aggregations on Selling_Price
print("Min selling price: ", selling_price.min())
print("Sum of selling prices: ", np.sum(selling_price))
print("Index of min selling price: ", np.argmin(selling_price))
print("Index of max selling price: ", np.argmax(selling_price))
print("Variance of selling price: ", np.var(selling_price))
print("Median of selling price: ", np.median(selling_price))
print("Mean of selling price: ", selling_price.mean())
print("Max selling price: ", selling_price.max())
print("50th percentile of selling price: ", np.percentile(selling_price, 50))


# Relational Operators
print("Relational Operations:")
print("Present Price < 5: ", present_price < 5)
print("Driven KMs <= 50000: ", driven_kms <= 50000)
print("Selling Price > 2: ", selling_price > 2)
print("Manufacturing Year >= 2015: ", manufacturing_year >= 2015)
print("Selling Price == 3: ", selling_price == 3)
print("Driven KMs != 10000: ", driven_kms != 10000)


# Masking
print("Adding 1 to Present Price:", present_price + 1)
print("Multiplying Driven KMs by 2:", driven_kms * 2)
print("Selecting Selling Price values greater than 3:", selling_price[selling_price > 3])


# Select 5 random rows using Fancy Indexing
random_indices = np.random.choice(data.index, 5, replace=False)
random_car_data = data.loc[random_indices]
print("Randomly selected car data:\n", random_car_data)


import matplotlib.pyplot as plt
# Group the data by Present Price ranges
bins = [0, 5, 10, 15, np.inf]
labels = ['Below 5', '5-10', '10-15', 'Above 15']
data['Present_Price_Range'] = pd.cut(data['Present_Price'], bins=bins, labels=labels, right=False)
# Calculate the average Selling Price for each Present Price range
avg_selling_price = data.groupby('Present_Price_Range', observed=False)['Selling_Price'].mean()  # Set observed=False
# Create a bar graph
avg_selling_price.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Average Selling Price by Present Price Range')
plt.xlabel('Present Price Range (Lakhs)')
plt.ylabel('Average Selling Price (Lakhs)')
# Show the plot
plt.show()


import matplotlib.pyplot as plt
# Group the data by Present Price ranges
bins = [0, 5, 10, 15, np.inf]
labels = ['Below 5', '5-10', '10-15', 'Above 15']
data['Present_Price_Range'] = pd.cut(data['Present_Price'], bins=bins, labels=labels, right=False)
# Calculate the average Selling Price for each Present Price range
avg_selling_price = data.groupby('Present_Price_Range', observed=False)['Selling_Price'].mean()  # Set observed=False
# Plot a pie chart
plt.pie(avg_selling_price, labels=avg_selling_price.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Average Selling Price Distribution by Present Price Range')
# Show the plot
plt.show()

# Extract the 'Selling_Price'
selling_price = np.array(data['Selling_Price'])
# Create a vector of length 40 with values between the min and max of 'Selling_Price'
a = np.linspace(selling_price.min(), selling_price.max(), num=40)
# Print the vector
print("Vector of 40 evenly spaced values between min and max of Selling Price:")
print(a)


# Extract the 'Selling_Price'
selling_price = np.array(data['Selling_Price'])
# Normalize the selling price data between 0 and 1
selling_price_normalized = (selling_price - selling_price.min()) / (selling_price.max() - selling_price.min())
# Create a vector of length 40 with values between the min and max of normalized 'Selling_Price'
a = np.linspace(selling_price_normalized.min(), selling_price_normalized.max(), num=40)
# Print the original range and the normalized vector
print("Original Selling Price range: [", selling_price.min(), ",", selling_price.max(), "]")
print("Vector of 40 evenly spaced normalized values between 0 and 1:")
print(a)


#Join operation
# Select 'Present_Price' and 'Selling_Price'
data_present_price = data[['Present_Price']].set_index(data.index)
data_selling_price = data[['Selling_Price']].set_index(data.index)
# Print the dataframes
print("Present Price DataFrame:")
print(data_present_price, end="\n\n")
print("Selling Price DataFrame:")
print(data_selling_price, end="\n\n")
# Perform the join operation
joined_data = data_present_price.join(data_selling_price, lsuffix="_Present_Price", rsuffix="_Selling_Price")
print("Joined DataFrame:")
print(joined_data)


# Pivot table
# Create a pivot table
pivot_table = data.pivot_table(values='Selling_Price', index='Present_Price', aggfunc='mean')
# Print the pivot table
print("Pivot table - Average Selling Price by Present Price:",pivot_table)


# Swap rows in reverse order
reversed_rows = data.iloc[::-1]
# Swap columns in reverse order
reversed_columns = data.iloc[:, ::-1]
# Swap both rows and columns in reverse order
reversed_both = data.iloc[::-1, ::-1]
print("Rows swapped in reverse order:",reversed_rows)
print("\nColumns swapped in reverse order:",reversed_columns)
print("\nBoth rows and columns swapped in reverse order:",reversed_both)


# String operations on the 'Car_Name_str' column
# Convert 'Car_Name' column values to string
data['Car_Name_str'] = data['Car_Name'].astype(str)
# Converts the string to lowercase.
print("Lowercase:\n", data['Car_Name_str'].str.lower())


# Finds the length of each string.
print("\nLength:\n", data['Car_Name_str'].str.len())


# Checks if the string starts with "A"
print("\nStarts with 'A':\n", data['Car_Name_str'].str.startswith('A'))

# Splits the string into a list
print("\nSplit:\n", data['Car_Name_str'].str.split())


# extracts the first letter or digits
print("\nExtract first digits or letters:\n", data['Car_Name_str'].str.extract('([A-Za-z]+)', expand=False))


# Finds all instances of a regex pattern in the string.
# In this case, it checks patterns that don't start or end with vowels.
print("\nFind patterns that don't start/end with vowels:\n", data['Car_Name_str'].str.findall(r'^[^AEIOU].*[^aeiou]$'))

# Gets the first 3 characters
print("\nFirst 3 characters:\n", data['Car_Name_str'].str[0:3])

# Gets the last element after splitting the string
print("\nLast split element:\n", data['Car_Name_str'].str.split().str.get(-1))

# Analysis of Present_Price, Selling_Price, Driven_kms
columns = ['Present_Price', 'Selling_Price', 'Driven_kms']
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


# Sort the column of 'Selling_Price'
sorted_data = data.sort_values(by='Selling_Price')
# Print sorted 'Selling_Price'
print("Sorted Dataset by Selling_Price:")
print(sorted_data[['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Driven_kms']])


# Handling Missing Data
print("\nMissing Values Before Handling:")
print(data.isnull().sum())

# fill missing values with the mode
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
print("\nMissing Values After Handling:")
print(data.isnull().sum())

# Data Filtering
filtered_data = data[data['Selling_Price'] > 1000]
print("\nFiltered Data (Selling_Price > 1000):",filtered_data.head())


# Data Transformation
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Identify numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
# Ensure there are numeric columns before proceeding
if len(numeric_columns) > 0:
    print("Numeric columns detected:", numeric_columns)

    scaler = StandardScaler()

    # Convert to numeric, NaN if invalid
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # Fill NaN with mean for scaling
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Perform scaling
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    print("\nData After Transformation (Scaling):",data.head())
else:
    print("No numeric columns found in the dataset.")


import pandas as pd
data = pd.DataFrame({
    'Make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
    'Model': ['Corolla', 'Civic', 'Focus', 'X5', 'A4'],
    'Year': [2015, 2018, 2016, 2020, 2017],
    'Price': [15000, 18000, 12000, 35000, 27000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'Mileage': [15, 20, 18, 12, 16]
})
# Check the data types before conversion
print("Data Types Before Conversion:",data.dtypes)
# Convert columns to appropriate data types
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Fuel_Type'] = data['Fuel_Type'].astype('category')
data['Year'] = data['Year'].astype('int64')
data['Price'] = data['Price'].astype('float64')
data['Mileage'] = data['Mileage'].astype('float64')
# Check the data types after conversion
print("\nData Types After Conversion:",data.dtypes)
# Print the modified dataset
print("\nModified Dataset:",data)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load your dataset
data = pd.read_csv(r"car_price_prediction_ML.csv")

# Check the first few rows and column names
print(data.head())
print(data.columns)

# Check if 'Present_Price' column exists in the dataset
if 'Present_Price' in data.columns:
    X = data.drop('Present_Price', axis=1)
    y = data['Present_Price']
else:
    print("Column 'Present_Price' not found in dataset.")

if 'Present_Price' in data.columns:

    # Step 1: Handle categorical features using OneHotEncoding and scaling
    categorical_columns = X.select_dtypes(include=['object']).columns
    # Define the preprocessor for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ])

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Scaling and Encoding
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Step 4: Train a linear regression model
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)

    # Step 5: Make predictions
    y_pred = model.predict(X_test_transformed)

    # Step 6: Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(f'R-squared (R2) Score: {r2}')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load your dataset
data = pd.read_csv(r"car_price_prediction_ML.csv")

# Check the first few rows and column
print(data.head())
print(data.columns)

# Check if 'Selling_Price' column exists
if 'Selling_Price' in data.columns:
    X = data.drop('Selling_Price', axis=1)
    y = data['Selling_Price']
else:
    print("Column 'Selling_Price' not found in dataset.")

# Handle categorical features and apply transformations
categorical_columns = X.select_dtypes(include=['object']).columns

# Define the preprocessor for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),  # Scale numeric features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # Encode categorical features
    ])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply transformations (scaling and encoding)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Step 4: Train a linear regression model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test_transformed)

# Step 6: Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2) Score: {r2}')

# Predict the features
swift_data = data[data['Car_Name'] == 'swift']
swift_features = swift_data.drop(['Selling_Price'], axis=1)

# Apply transformations
swift_features_transformed = preprocessor.transform(swift_features)

# Predict Selling_Price for 'swift' car
predicted_selling_price = model.predict(swift_features_transformed)
print(f"Predicted Selling Price for 'swift': {predicted_selling_price[0]}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv(r"car_price_prediction_ML.csv")

# Check for required columns
if 'Selling_Price' in data.columns and 'Year' in data.columns:
    X = data.drop('Selling_Price', axis=1)
    y = data['Selling_Price']
else:
    raise ValueError("Dataset must contain 'Selling_Price' and 'Year' columns.")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2) Score: {r2}')

# Specific prediction for 'ritz'
ritz_data = data[data['Car_Name'] == 'ritz']
if not ritz_data.empty:
    ritz_features = ritz_data.drop(['Selling_Price'], axis=1)
    ritz_transformed = preprocessor.transform(ritz_features)
    predicted_selling_price = model.predict(ritz_transformed)
    for year, price in zip(ritz_data['Year'], predicted_selling_price):
        print(f"Predicted Selling Price for 'ritz' ({year}): {price}")
else:
    print("'ritz' not found in the dataset.")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv(r"car_price_prediction_ML.csv")

# Check for required columns
if 'Selling_Price' in data.columns and 'Year' in data.columns:
    X = data.drop('Selling_Price', axis=1)
    y = data['Selling_Price']
else:
    raise ValueError("Dataset must contain 'Selling_Price' and 'Year' columns.")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2) Score: {r2}')

# Specific prediction for 'swift'
swift_data = data[data['Car_Name'] == 'swift']
if not swift_data.empty:
    swift_features = swift_data.drop(['Selling_Price'], axis=1)
    swift_transformed = preprocessor.transform(swift_features)
    predicted_selling_price = model.predict(swift_transformed)
    for year, price in zip(swift_data['Year'], predicted_selling_price):
        print(f"Predicted Selling Price for 'swift' ({year}): {price}")
else:
    print("'swift' not found in the dataset.")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv(r"car_price_prediction_ML.csv")

# Check for required columns
if 'Selling_Price' in data.columns and 'Year' in data.columns:
    X = data.drop('Selling_Price', axis=1)
    y = data['Selling_Price']
else:
    raise ValueError("Dataset must contain 'Selling_Price' and 'Year' columns.")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Evaluate the model
y_pred = model.predict(X_test_transformed)
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2) Score: {r2}')

# Specific prediction for 'ciaz'
ciaz_data = data[data['Car_Name'] == 'ciaz']
if not ciaz_data.empty:
    ciaz_features = ciaz_data.drop(['Selling_Price'], axis=1)
    ciaz_transformed = preprocessor.transform(ciaz_features)
    predicted_selling_price = model.predict(ciaz_transformed)
    for year, price in zip(ciaz_data['Year'], predicted_selling_price):
        print(f"Predicted Selling Price for 'ciaz' ({year}): {price}")
else:
    print("'ciaz' not found in the dataset.")
