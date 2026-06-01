# Project 2: Simple Linear Regression on Housing Prices

![HEX SOFTWARES](https://img.shields.io/badge/HEX-SOFTWARES-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Machine Learning](https://img.shields.io/badge/ML-Linear%20Regression-orange)

## 🎯 Project Overview

This project implements a **Simple Linear Regression model** to predict housing prices based on various property features using the Boston Housing dataset. It includes comprehensive data preprocessing, feature selection, normalization, and model evaluation.  

## 📋 Task Description

- **Use** a housing prices dataset (Boston Housing dataset)
- **Perform** data preprocessing, including feature selection and normalization
- **Build** a linear regression model to predict house prices based on features like:
  - Number of rooms
  - Square footage 
  - Crime rate
  - Property tax
  - And more...   

## 🗂️ Project Structure

```
Project-2-Housing-Price-Prediction/
│
├── housing_price_prediction.py    # Main Python script
├── housing_price_analysis.png     # Comprehensive visualizations
├── prediction_results.csv         # Model predictions
├── model_summary.txt              # Model coefficients and metrics
├── PROJECT_REPORT.txt             # Detailed project report
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Project

```bash
python housing_price_prediction.py
```

### Expected Output

The script will:
1. ✅ Load the housing dataset
2. ✅ Perform exploratory data analysis
3. ✅ Select relevant features
4. ✅ Preprocess and normalize data
5. ✅ Train a Linear Regression model
6. ✅ Evaluate performance
7. ✅ Generate visualizations
8. ✅ Save results

## 📊 Key Results

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **R² Score** | 0.2440 | 0.2210 |
| **RMSE** | 1.3008 | 1.6196 |
| **MAE** | 0.7968 | 0.8414 |

## 🔍 Features Used

The model uses 10 carefully selected features:

1. **RM** - Average number of rooms per dwelling
2. **LSTAT** - % lower status of the population
3. **PTRATIO** - Pupil-teacher ratio
4. **RAD** - Index of accessibility to highways
5. **AGE** - Proportion of owner-occupied units built before 1940
6. **NOX** - Nitric oxides concentration
7. **INDUS** - Proportion of non-retail business acres
8. **ZN** - Proportion of residential land zoned
9. **DIS** - Distances to employment centers
10. **CHAS** - Charles River dummy variable

## 📈 Visualizations

The project generates 9 comprehensive visualizations:

1. 🔥 **Correlation Heatmap** - Feature relationships
2. 📊 **Price Distribution** - Target variable analysis
3. 📍 **Actual vs Predicted (Training)** - Model fit
4. 📍 **Actual vs Predicted (Test)** - Generalization
5. 📉 **Residual Plot (Training)** - Error patterns
6. 📉 **Residual Plot (Test)** - Error validation
7. 🎯 **Feature Importance** - Coefficient magnitudes
8. 📊 **Residual Distribution** - Error normality
9. 📊 **Performance Comparison** - Metrics comparison

## 🔬 Methodology

### 1. Data Loading
- Dataset: Boston Housing (synthetic)
- Samples: 506 properties
- Features: 13 characteristics

### 2. Data Preprocessing
- Missing value check
- Outlier detection using IQR
- Feature selection via correlation
- Train-test split (80-20)

### 3. Feature Normalization
- Method: StandardScaler
- Mean: 0
- Standard Deviation: 1

### 4. Model Training
- Algorithm: Linear Regression (OLS)
- Optimization: Least Squares

### 5. Evaluation
- Metrics: R², MSE, RMSE, MAE
- Validation: Train-test comparison

## 💡 Key Insights

### Positive Predictors (↑ Price)
- 🏠 **Number of rooms** (+0.498)
- 🌳 **Residential zoning** (+0.155)
- 🚗 **Distance to employment** (+0.156)

### Negative Predictors (↓ Price)
- 👥 **Lower status population** (-0.350)
- 🎓 **Pupil-teacher ratio** (-0.212)
- 🚦 **Highway accessibility** (-0.100)

## 📝 Code Walkthrough

```python
# Step 1: Load Data
df = pd.DataFrame(...)  # Housing dataset

# Step 2: Feature Selection
correlation = df.corr()['PRICE']
selected_features = correlation[abs(correlation) > 0.05]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 5: Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
```

## 🎓 Learning Outcomes

By completing this project, you will understand:

- ✅ Data preprocessing techniques
- ✅ Feature selection methods
- ✅ Feature normalization/standardization
- ✅ Linear regression implementation
- ✅ Model evaluation metrics
- ✅ Visualization of results
- ✅ Interpretation of coefficients

## 🔄 Future Improvements

1. **Feature Engineering**
   - Add interaction terms
   - Create polynomial features
   - Domain-specific features

2. **Advanced Models**
   - Ridge/Lasso regression
   - Random Forest
   - Gradient Boosting
   - Neural Networks

3. **Validation**
   - K-fold cross-validation
   - Stratified sampling
   - Time-based splits

4. **Deployment**
   - Flask/FastAPI web service
   - Streamlit dashboard
   - Docker containerization

## 📚 References

1. Harrison & Rubinfeld (1978) - Original Boston Housing study
2. James et al. (2013) - "An Introduction to Statistical Learning"
3. Scikit-learn Documentation

## 👤 Author

**[Your Name]**
- Internship: HEX SOFTWARES
- Date: February 2, 2026
- Project: Housing Price Prediction using Linear Regression

## 📄 License

This project is created for educational and internship purposes.

## 🤝 Acknowledgments

Special thanks to **HEX SOFTWARES** for providing this learning opportunity.

---

**Note**: This is an internship project demonstrating understanding of linear regression, data preprocessing, and machine learning workflows.

## 📧 Contact

For questions or discussions about this project, feel free to reach out!

---

Made with ❤️ for learning and growth in Data Science
