import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Step 1: Extract - Load the dataset
try:
    df = pd.read_csv("input_data.csv")
    print("Data loaded successfully!\n", df.head())
except FileNotFoundError:
    print("Error: input_data.csv not found!")
    exit()

# Step 2: Handle Missing Data
imputer = SimpleImputer(strategy="mean")  # Replace missing values with mean
df["Age"] = imputer.fit_transform(df[["Age"]])
df["Salary"] = imputer.fit_transform(df[["Salary"]])

# Step 3: Encode Categorical Data
encoder = LabelEncoder()
df["Department"] = encoder.fit_transform(df["Department"])

# Step 4: Normalize Numerical Data
scaler = StandardScaler()
df[["Age", "Salary"]] = scaler.fit_transform(df[["Age", "Salary"]])

# Step 5: Save the Transformed Data
df.to_csv("processed_data.csv", index=False)
print("\nData processing complete. Processed data saved as 'processed_data.csv'.")
print(df.head())  # Display the first few rows