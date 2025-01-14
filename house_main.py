import streamlit as st
import pandas as pd
import house_home
import house_models
import house_plots
from sklearn.datasets import fetch_california_housing
# Configure your house page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title='House Price Prediction',
                   page_icon='random',
                   layout='wide',
                   initial_sidebar_state='auto'
                   )


# Loading the dataset.
@st.cache_data()
def load_data():


    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target
    return df

house_df = load_data()

st.title('California Housing Prices Prediction')
pages_dict = {"house": house_home,
              "Rrediction": house_models,
              "Plots": house_plots}

st.sidebar.title('Navigation')
user_choice = st.sidebar.radio('Go To', tuple(pages_dict.keys()))
selected_page = pages_dict[user_choice]
selected_page.app(house_df)