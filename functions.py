import pandas as pd
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import requests
import warnings
import os

warnings.filterwarnings("ignore")

@st.cache_data
def load_dfs():
    url1 = 'https://raw.githubusercontent.com/OlgaGnezdilova/Health-Calculator/main/df_food.csv'
    url2 = 'https://raw.githubusercontent.com/OlgaGnezdilova/Health-Calculator/main/recommendations.csv'
    df_food = pd.read_csv(url1)
    df_recomm = pd.read_csv(url2)
    return df_food, df_recomm

@st.cache_data
def load_model():
    model_url = 'https://github.com/OlgaGnezdilova/Health-Calculator/raw/main/linear_regression_model.pkl'
    local_model_path = 'linear_regression_model.pkl'
    
    # Download the model if it's not already downloaded
    if not os.path.exists(local_model_path):
        response = requests.get(model_url)
        with open(local_model_path, 'wb') as f:
            f.write(response.content)
    
    model = joblib.load(local_model_path)
    return model

def calculate_daily_calories(gender, weight, height, age, activity_factor):
    if gender.lower() == 'm':
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) + 5
    elif gender.lower() == 'f':
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) - 161
    else:
        raise ValueError("Invalid gender. Use 'm' or 'f'.")
    return bmr * float(activity_factor)

def predict_calories(model, gender_encoded, age, height, weight, duration, pulse):
    user_input = np.array([[float(gender_encoded), float(age), float(height), float(weight), float(duration), float(pulse)]])
    return model.predict(user_input)

def display_food_suggestions(df_food, df_consumed, difference):
    df_difference = df_food[df_food['Calories'] < -difference][['Food', 'Calories', 'Sugar']]
    if not df_difference.empty:
        df_difference_no_index = df_difference.reset_index(drop=True)
        df_difference_no_index.iloc[:, -1] = df_difference_no_index.iloc[:, -1].apply(lambda x: f'{x:.2f}')
        df_difference_no_index.iloc[:, -2] = df_difference_no_index.iloc[:, -2].apply(lambda x: f'{x:.2f}')
        df_difference_no_index.index += 1
        st.write(df_difference_no_index.sample(10))
    else:
        st.write("No suggestions available. You're already meeting or exceeding your daily calorie goal!")

def generate_top_foods(df_food, nutrient, nutrient_name):
    st.subheader(f"Here is Top 10 of {nutrient_name}-rich food:")
    top_foods = df_food.nlargest(10, nutrient)[['Food', nutrient, 'Sugar', 'Carbohydrate']]
    for col in ['Sugar', 'Carbohydrate']:
        top_foods[col] = top_foods[col].apply(lambda x: f'{x:.2f}')
    top_foods_no_index = top_foods.reset_index(drop=True)
    top_foods_no_index.index += 1
    st.write(top_foods_no_index)




