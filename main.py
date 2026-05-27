# ==========================================
# House Price Prediction using Linear Regression
# California Housing Dataset
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# ==========================================
# 1. Load Dataset
# ==========================================

housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target column
df['Price'] = housing.target

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# ==========================================
# 2. Dataset Information
# ==========================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# 3. Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 4. Correlation Heatmap
# ==========================================

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# 5. Distribution of House Prices
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()

# ==========================================
# 6. Feature Selection
# ==========================================

X = df.drop('Price', axis=1)
y = df['Price']

# ==========================================
# 7. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# 8. Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# ==========================================
# 9. Make Predictions
# ==========================================

y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])

# ==========================================
# 10. Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("----------------------------")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# ==========================================
# 11. Actual vs Predicted Plot
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# ==========================================
# 12. Feature Coefficients
# ==========================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients:")
print(coefficients)

# ==========================================
# 13. Save the Model
# ==========================================

joblib.dump(model, 'house_price_model.pkl')

print("\nModel saved as 'house_price_model.pkl'")

# ==========================================
# 14. Final Conclusion
# ==========================================

print("\nProject Completed Successfully!")
print("Linear Regression Model Built and Evaluated.")