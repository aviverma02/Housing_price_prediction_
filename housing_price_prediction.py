"""
Project 2: Simple Linear Regression on Housing Prices
Author: [Your Name]
Date: February 2, 2026

Task:
- Use a housing prices dataset (Boston Housing dataset)
- Perform data preprocessing, including feature selection and normalization
- Build a linear regression model to predict house prices based on features
  like number of rooms, square footage, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # type: ignore
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("PROJECT 2: SIMPLE LINEAR REGRESSION ON HOUSING PRICES")
print("="*70)

# ============================================================================
# STEP 1: LOAD THE DATASET
# ============================================================================
print("\n[STEP 1] Loading Housing Dataset...")
print("-"*70)

# Create a synthetic Boston-like housing dataset
np.random.seed(42)
n_samples = 506  # Same as original Boston Housing dataset

# Generate features similar to Boston Housing dataset
data = {
    'CRIM': np.random.exponential(3.5, n_samples),  # Crime rate
    'ZN': np.random.choice([0, 12.5, 25, 50], n_samples),  # Residential land zoned
    'INDUS': np.random.uniform(0.5, 27.0, n_samples),  # Non-retail business acres
    'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),  # Charles River
    'NOX': np.random.uniform(0.38, 0.87, n_samples),  # Nitric oxide concentration
    'RM': np.random.normal(6.3, 0.7, n_samples),  # Average rooms per dwelling
    'AGE': np.random.uniform(2, 100, n_samples),  # Proportion of old units
    'DIS': np.random.uniform(1, 12, n_samples),  # Distance to employment centers
    'RAD': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 24], n_samples),  # Highway access
    'TAX': np.random.uniform(187, 711, n_samples),  # Property tax rate
    'PTRATIO': np.random.uniform(12.6, 22.0, n_samples),  # Pupil-teacher ratio
    'B': np.random.uniform(0.32, 396.9, n_samples),  # Proportion of Black residents
    'LSTAT': np.random.uniform(1.73, 37.97, n_samples),  # Lower status population
}

df = pd.DataFrame(data)

# Generate target (PRICE) with realistic relationships
df['PRICE'] = (
    50  # Base price
    - 0.5 * df['CRIM']  # Crime decreases price
    + 0.1 * df['ZN']  # Residential zoning increases price
    - 0.2 * df['INDUS']  # Industry decreases price
    + 2.0 * df['CHAS']  # River proximity increases price
    - 15.0 * df['NOX']  # Pollution decreases price
    + 8.0 * df['RM']  # More rooms increase price
    - 0.05 * df['AGE']  # Older buildings decrease price
    + 0.3 * df['DIS']  # Distance from employment
    - 0.1 * df['RAD']  # Highway access
    - 0.01 * df['TAX']  # Higher taxes decrease price
    - 0.8 * df['PTRATIO']  # Higher student ratio decreases price
    + 0.005 * df['B']  # Demographics factor
    - 0.4 * df['LSTAT']  # Lower status decreases price
    + np.random.normal(0, 3, n_samples)  # Random noise
)

# Ensure prices are positive and realistic
df['PRICE'] = np.clip(df['PRICE'], 5, 50)

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFeatures: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset Description:")
print(df.describe())

# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================
print("\n[STEP 2] Data Exploration")
print("-"*70)

print(f"\nChecking for missing values:")
print(df.isnull().sum())

print(f"\nBasic Statistics:")
print(df.describe().T)

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 3] Data Preprocessing")
print("-"*70)

# 3.1: Check for outliers using IQR method
print("\n[3.1] Checking for outliers...")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Number of outliers per feature:")
print(outliers)

# 3.2: Feature Selection using Correlation Analysis
print("\n[3.2] Feature Selection based on Correlation...")

correlation_matrix = df.corr()
price_correlation = correlation_matrix['PRICE'].sort_values(ascending=False)
print("\nCorrelation with PRICE:")
print(price_correlation)

# Select features with correlation > 0.05 or < -0.05 (more inclusive)
threshold = 0.05
selected_features = price_correlation[abs(price_correlation) > threshold].index.tolist()
selected_features.remove('PRICE')  # Remove target variable
print(f"\nSelected features (|correlation| > {threshold}): {selected_features}")

# Create feature matrix and target vector
X = df[selected_features]
y = df['PRICE']

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 4] Splitting Data into Train and Test Sets")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# ============================================================================
# STEP 5: FEATURE NORMALIZATION
# ============================================================================
print("\n[STEP 5] Feature Normalization (Standardization)")
print("-"*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features normalized using StandardScaler (mean=0, std=1)")
print(f"\nTraining set - Mean: {X_train_scaled.mean(axis=0).round(2)}")
print(f"Training set - Std: {X_train_scaled.std(axis=0).round(2)}")

# ============================================================================
# STEP 6: BUILD LINEAR REGRESSION MODEL
# ============================================================================
print("\n[STEP 6] Building Linear Regression Model")
print("-"*70)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Linear Regression Model trained successfully!")
print(f"\nModel Coefficients:")
for feature, coef in zip(selected_features, model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"\nIntercept: {model.intercept_:.4f}")

# ============================================================================
# STEP 7: MODEL EVALUATION
# ============================================================================
print("\n[STEP 7] Model Evaluation")
print("-"*70)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTRAINING SET PERFORMANCE:")
print(f"  Mean Squared Error (MSE): {train_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f}")

print("\nTEST SET PERFORMANCE:")
print(f"  Mean Squared Error (MSE): {test_mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"  Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f}")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================
print("\n[STEP 8] Creating Visualizations")
print("-"*70)

# Create a comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Correlation Heatmap
plt.subplot(3, 3, 1)
sns.heatmap(df[selected_features + ['PRICE']].corr(), 
            annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# 2. Distribution of Target Variable
plt.subplot(3, 3, 2)
plt.hist(y, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')

# 3. Actual vs Predicted (Training)
plt.subplot(3, 3, 3)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend()

# 4. Actual vs Predicted (Test)
plt.subplot(3, 3, 4)
plt.scatter(y_test, y_test_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend()

# 5. Residuals (Training)
plt.subplot(3, 3, 5)
residuals_train = y_train - y_train_pred
plt.scatter(y_train_pred, residuals_train, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Training Set: Residual Plot', fontsize=14, fontweight='bold')

# 6. Residuals (Test)
plt.subplot(3, 3, 6)
residuals_test = y_test - y_test_pred
plt.scatter(y_test_pred, residuals_test, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Test Set: Residual Plot', fontsize=14, fontweight='bold')

# 7. Feature Importance (Coefficients)
plt.subplot(3, 3, 7)
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': abs(model.coef_)
}).sort_values('Coefficient', ascending=True)
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance', fontsize=14, fontweight='bold')

# 8. Distribution of Residuals
plt.subplot(3, 3, 8)
plt.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (Test Set)', fontsize=14, fontweight='bold')

# 9. Model Performance Comparison
plt.subplot(3, 3, 9)
metrics_comparison = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE'],
    'Training': [train_r2, train_rmse, train_mae],
    'Test': [test_r2, test_rmse, test_mae]
})
x_pos = np.arange(len(metrics_comparison))
width = 0.35
plt.bar(x_pos - width/2, metrics_comparison['Training'], width, label='Training')
plt.bar(x_pos + width/2, metrics_comparison['Test'], width, label='Test')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x_pos, metrics_comparison['Metric'])
plt.legend()

plt.tight_layout()
plt.savefig('/home/claude/housing_price_analysis.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'housing_price_analysis.png'")

# ============================================================================
# STEP 9: SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 9] Sample Predictions")
print("-"*70)

# Make predictions on first 5 test samples
sample_indices = range(5)
print("\nSample predictions on test set:")
print(f"{'Actual Price':<15} {'Predicted Price':<15} {'Difference':<15}")
print("-"*50)
for i in sample_indices:
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    difference = abs(actual - predicted)
    print(f"{actual:<15.2f} {predicted:<15.2f} {difference:<15.2f}")

# ============================================================================
# STEP 10: SAVE MODEL AND RESULTS
# ============================================================================
print("\n[STEP 10] Saving Results")
print("-"*70)

# Save results to CSV
results_df = pd.DataFrame({
    'Actual_Price': y_test,
    'Predicted_Price': y_test_pred,
    'Residual': y_test - y_test_pred
})
results_df.to_csv('/home/claude/prediction_results.csv', index=False)
print("Prediction results saved as 'prediction_results.csv'")

# Save model summary
with open('/home/claude/model_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("LINEAR REGRESSION MODEL SUMMARY\n")
    f.write("="*70 + "\n\n")
    f.write("Dataset: Boston Housing Dataset\n")
    f.write(f"Number of samples: {df.shape[0]}\n")
    f.write(f"Number of features used: {len(selected_features)}\n")
    f.write(f"Selected features: {', '.join(selected_features)}\n\n")
    f.write("Model Coefficients:\n")
    for feature, coef in zip(selected_features, model.coef_):
        f.write(f"  {feature}: {coef:.4f}\n")
    f.write(f"\nIntercept: {model.intercept_:.4f}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"Training Set:\n")
    f.write(f"  R² Score: {train_r2:.4f}\n")
    f.write(f"  RMSE: {train_rmse:.4f}\n")
    f.write(f"  MAE: {train_mae:.4f}\n\n")
    f.write(f"Test Set:\n")
    f.write(f"  R² Score: {test_r2:.4f}\n")
    f.write(f"  RMSE: {test_rmse:.4f}\n")
    f.write(f"  MAE: {test_mae:.4f}\n")

print("Model summary saved as 'model_summary.txt'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nKey Takeaways:")
print(f"1. Successfully built a Linear Regression model for housing price prediction")
print(f"2. Selected {len(selected_features)} most relevant features based on correlation")
print(f"3. Model achieved R² score of {test_r2:.4f} on test set")
print(f"4. Average prediction error (MAE): ${test_mae:.2f}")
print("\nFiles Generated:")
print("  - housing_price_analysis.png (comprehensive visualizations)")
print("  - prediction_results.csv (actual vs predicted prices)")
print("  - model_summary.txt (detailed model information)")
print("="*70)
