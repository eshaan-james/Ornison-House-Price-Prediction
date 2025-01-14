# California House Price Prediction

## Overview

This project demonstrates the use of various regression techniques to predict housing prices in California based on features from a housing dataset. The models used include **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and **Gradient Boosting Regressor**. Their performance was compared to identify the best-performing model.

## Features

- **Dataset Used:** California housing dataset containing the following attributes:
  - **MedInc**: Median income in the block group.
  - **HouseAge**: Median age of the houses in the block group.
  - **AveRooms**: Average number of rooms per household.
  - **AveBedrms**: Average number of bedrooms per household.
  - **Population**: Population in the block group.
  - **AveOccup**: Average number of occupants per household.
  - **Latitude**: Latitude coordinate of the block group.
  - **Longitude**: Longitude coordinate of the block group.

- **Objective:** Predict the **Median House Value** for each block group based on the input features.

- **Technologies Used:**
  - **Python Libraries**:
    - `pandas` and `numpy` for data preprocessing and manipulation.
    - `scikit-learn` for model implementation and evaluation.
    - `matplotlib` and `seaborn` for data visualization.

## How It Works

1. **Data Preprocessing**:
   - Cleaned the dataset by handling missing values and scaling the features.
   - Performed feature engineering to ensure meaningful inputs for the models.

2. **Exploratory Data Analysis (EDA):**
   - Visualized the distribution of features and their relationship with the target variable.
   - Examined correlations between features to identify multicollinearity.

3. **Model Building and Evaluation:**
   - Implemented the following regression models using `scikit-learn`:
     - **Linear Regression**: A simple model to establish a baseline.
     - **Decision Tree Regressor**: To capture non-linear patterns in the data.
     - **Random Forest Regressor**: An ensemble method to improve accuracy by combining multiple trees.
     - **Gradient Boosting Regressor**: Another ensemble technique optimized for lower bias.
   - Split the dataset into training and testing subsets to evaluate the models’ performance.
   - Metrics used for evaluation:
     - **Test Accuracy**
     - **Root Mean Squared Error (RMSE)**

4. **Model Comparison:**
   - Compared the performance of all four models to identify the most accurate one.
   - Visualized the results using bar plots and error analysis charts.

## Results

- The models were evaluated based on their predictive accuracy and error metrics.
- Observations were made regarding the suitability of each model for the given dataset.

## Future Work

- Explore additional advanced models such as **XGBoost** or **CatBoost**.
- Incorporate hyperparameter tuning to optimize model performance.
- Deploy the model using a web-based interface like **Streamlit** or **Flask** for real-world applications.

