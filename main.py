# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
housing = fetch_california_housing(as_frame=True)

# Extract the feature variables (X) and target variable (y)
X = housing.data          # Features like income, rooms, etc.
y = housing.target        # House price (in $100,000s)
y = y.rename("PRICE")     # Rename for clarity

# Show the first 5 rows of features
print("🧾 Features (first 5 rows):")
print(X.head())

# Show the first 5 target values
print("\n💰 House Prices (first 5 values):")
print(y.head())


# Step 3: Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Combine features and target into one DataFrame for easy analysis
data = pd.concat([X, y], axis=1)

# 1. Basic information
print("\n🧠 Dataset Info:")
print(data.info())

# 2. Summary statistics
print("\n📊 Summary Statistics:")
print(data.describe())

# 3. Check for missing values
print("\n🔍 Missing Values:")
print(data.isnull().sum())

# 4. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# 5. Distribution of house prices
plt.figure(figsize=(8, 5))
sns.histplot(y, kde=True, color='green')
plt.title("Distribution of House Prices")
plt.xlabel("Price ($100,000s)")
plt.ylabel("Count")
plt.show()


# 6. Scatter plot: Median Income vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['MedInc'], y=data['PRICE'], alpha=0.5)
plt.title("Price vs. Median Income")
plt.xlabel("Median Income")
plt.ylabel("House Price ($100,000s)")
plt.show()


# Step 4: Data Preprocessing & Splitting

from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shape of data splits
print("\n📦 Data Split Summary:")
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape:  {X_test.shape}")
print(f"Training Labels Shape:   {y_train.shape}")
print(f"Testing Labels Shape:    {y_test.shape}")


# Step 5: Train a Linear Regression Model

from sklearn.linear_model import LinearRegression

# Create the model
lr_model = LinearRegression()

# Train the model using the training data
lr_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr_model.predict(X_test)


# Step 6: Evaluate the Model
from sklearn.metrics import mean_squared_error, r2_score

# Calculate RMSE (Root Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# R² score: 1 = perfect, 0 = worst
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# bonus insight->

# Step 7: Check Feature Importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\n📈 Feature Importance (Coefficients):")
print(coefficients)


# Visualization of the prdeictions->

# Step 8: Scatter Plot of Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="purple")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# for saving the model datas->

import joblib

# Save model
joblib.dump(lr_model, "linear_model.pkl")

# Optional: save feature names to check during prediction
joblib.dump(X.columns.tolist(), "feature_names.pkl")
