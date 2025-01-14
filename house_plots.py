import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import house_models
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor 

def app(house_df):
    warnings.filterwarnings('ignore')
    st.title("Visualise the House Price Prediction Web app ")

    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(house_df.corr(),annot=True)  
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        st.pyplot()
    
    if st.button("Generate Comparison Plot"):
        X_train, X_test, y_train, y_test = house_models.data_split(house_df)
        lin_reg = LinearRegression()
        lin_reg.fit(X_train,y_train)
        y_pred_lin = lin_reg.predict(X_test)

        dtree_reg = DecisionTreeRegressor()
        dtree_reg.fit(X_train,y_train)
        y_pred_dtree = dtree_reg.predict(X_test)

        rf_reg = RandomForestRegressor()
        rf_reg.fit(X_train, y_train)
        y_pred_rfr = rf_reg.predict(X_test)

        gb_reg = GradientBoostingRegressor()
        gb_reg.fit(X_train, y_train)
        y_pred_gbr = gb_reg.predict(X_test)

        models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

        training_accuracy = [lin_reg.score(X_train, y_train),
                            dtree_reg.score(X_train, y_train),
                            rf_reg.score(X_train, y_train),
                            gb_reg.score(X_train, y_train) 
                            ]  
        testing_accuracy =  [lin_reg.score(X_test, y_test),
                            dtree_reg.score(X_test, y_test),
                            rf_reg.score(X_test, y_test),
                            gb_reg.score(X_test, y_test) 
                            ]  
        rmse_scores = [np.sqrt(mean_squared_error(y_test, y_pred_lin)),
                    np.sqrt(mean_squared_error(y_test, y_pred_dtree)),
                    np.sqrt(mean_squared_error(y_test, y_pred_rfr)),
                    np.sqrt(mean_squared_error(y_test, y_pred_gbr)),
                    ]          
        # Convert RMSE scores to negative for visualization (lower is better)
        neg_rmse_scores = [-x for x in rmse_scores]

        bar_width = 0.25
        x = np.arange(len(models))

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - bar_width, training_accuracy, width=bar_width, label="Training Accuracy (%)", color='skyblue')
        ax.bar(x, testing_accuracy, width=bar_width, label="Testing Accuracy (%)", color='orange')
        ax.bar(x + bar_width, neg_rmse_scores, width=bar_width, label="Negative RMSE", color='green')

        # Customize the plot
        ax.set_xlabel("Models")
        ax.set_ylabel("Performance Metrics")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        st.pyplot(fig)
        

