import numpy as np
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor 


@st.cache_data()
def data_split(house_df):
    X = house_df.iloc[:, :-1]
    y = house_df.Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
    return X_train, X_test, y_train, y_test

# Models 
@st.cache_data()
def lin_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes ):
    X_train, X_test, y_train, y_test = data_split(house_df)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Classifify house quality using the 'predict()' function.
    prediction = lin_reg.predict([[Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes]])
    prediction = prediction[0]
    score = round(lin_reg.score(X_test, y_test) * 100, 3)
    return prediction, score
@st.cache_data()
def d_tree_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes ):
    X_train, X_test, y_train, y_test = data_split(house_df)

    dtree_reg = DecisionTreeRegressor()
    dtree_reg.fit(X_train, y_train)


    # Classifify house quality using the 'predict()' function.
    prediction = dtree_reg.predict([[Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes]])
    prediction = prediction[0]
    score = round(dtree_reg.score(X_test, y_test) * 100, 3)
    return prediction, score
@st.cache_data()
def rfr_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes ):
    X_train, X_test, y_train, y_test = data_split(house_df)

    rf_reg = RandomForestRegressor(random_state= 42)
    params = {
    'max_depth': range(10 , 60 , 10),
    'n_estimators': range(25 , 100 , 25)
    }
    rf_reg = GridSearchCV(rf_reg,param_grid= params,cv= 5,n_jobs= -1,verbose=1)
    rf_reg.fit(X_train, y_train)

    # Classifify house quality using the 'predict()' function.
    prediction = rf_reg.predict([[Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes]])
    prediction = prediction[0]
    score = round(rf_reg.score(X_test, y_test) * 100, 3)
    return prediction, score
@st.cache_data()
def gb_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes ):
    X_train, X_test, y_train, y_test = data_split(house_df)

    gb_reg = GradientBoostingRegressor()
    gb_reg.fit(X_train, y_train)

    # Classifify house quality using the 'predict()' function.
    prediction = gb_reg.predict([[Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes]])
    prediction = prediction[0]
    score = round(gb_reg.score(X_test, y_test) * 100, 3)
    return prediction, score

def app(house_df):
    st.markdown(
        "<p style='color:red;font-size:25px'>This app uses <b> Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor</b> for House Price Prediction.",
        unsafe_allow_html=True)
    
    st.subheader("Select Values:")

    Med_Inc = st.slider('Select median income', float(house_df['MedInc'].min()), float(house_df['MedInc'].max()), 0.1)

    House_Age = st.slider('Select median house age', float(house_df['HouseAge'].min()),float(house_df['HouseAge'].max()), 0.1)

    Ave_Rooms = st.slider('Select average number of rooms', float(house_df['AveRooms'].min()), float(house_df['AveRooms'].max()),  )

    Ave_Bedrms = st.slider('Select average number of bedrooms', float(house_df['AveBedrms'].min()), float(house_df['AveBedrms'].max()), 0.1)

    population = st.slider('Select block group population', float(house_df['Population'].min()), float(house_df['Population'].max()), 0.1)
    
    Ave_Occup = st.slider('Select average number of household members', float(house_df['AveOccup'].min()), float(house_df['AveOccup'].max()), 0.1)

    Latitudes = st.slider('Select block group latitude', float(house_df['Latitude'].min()), float(house_df['Latitude'].max()), 0.1)

    Longitudes = st.slider('Select block group longitude', float(house_df['Longitude'].min()), float(house_df['Longitude'].max()), 0.1)

    st.subheader("Model Selection")

    predictor = st.selectbox("Select the Prefiction Model", ('Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'Gradient Boosting Regressor'))

    if predictor == 'Linear Regression':
        prediction, score = lin_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes)
        if st.button("Predict"):
           
           st.write(f"House Price: {prediction}")
           st.write(f"The accuracy score is {score}%""")

    elif predictor == 'Decision Tree Regressor':
        prediction, score = d_tree_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes)
        if st.button("Predict"):
           
           st.write(f"Price of house: {prediction}")
           st.write(f"The accuracy score is {score}%""")

    elif predictor == 'Random Forest Regressor':
        prediction, score = rfr_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes)
        if st.button("Predict"):
           
           st.write(f"Price of house: {prediction}")
           st.write(f"The accuracy score is {score}%""")

    elif predictor == 'Gradient Boosting Regressor':
        prediction, score = gb_pred(house_df, Med_Inc, House_Age, Ave_Rooms, Ave_Bedrms, population, Ave_Occup, Latitudes, Longitudes)
        if st.button("Predict"):
           
           st.write(f"Price of house: {prediction}")
           st.write(f"The accuracy score is {score}%""")