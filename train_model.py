import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Create directories for static files if they don't exist
os.makedirs('static/images', exist_ok=True)

# Load the dataset
df = pd.read_csv('water_data.csv')

# Exploratory Data Analysis
print("Dataset Information:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation Analysis
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png')

# Feature Importance Analysis
X = df.drop(['Water_Requirement_TMC', 'Year'], axis=1)
y = df['Water_Requirement_TMC']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/images/feature_importance.png')

# Model Evaluation
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Create historical data visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['Water_Requirement_TMC'], marker='o', linestyle='-')
plt.title('Historical Water Requirement (1990-2019)')
plt.xlabel('Year')
plt.ylabel('Water Requirement (TMC)')
plt.grid(True)
plt.tight_layout()
plt.savefig('static/images/historical.png')

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("\nModel saved as 'model.pkl'")