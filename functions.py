import pandas as pd
import joblib
import numpy as np
import requests
import os

def load_dfs():
    url1 = 'https://raw.githubusercontent.com/OlgaGnezdilova/Health-Calculator/main/df_food.csv'
    url2 = 'https://raw.githubusercontent.com/OlgaGnezdilova/Health-Calculator/main/recommendations.csv'
    df_food = pd.read_csv(url1)
    df_recomm = pd.read_csv(url2)
    return df_food, df_recomm

def load_model():
    model_url = 'https://github.com/OlgaGnezdilova/Health-Calculator/raw/main/linear_regression_model.pkl'
    model_filename = model_url.split('/')[-1]
    
    if not os.path.exists(model_filename):
        r = requests.get(model_url, allow_redirects=True)
        with open(model_filename, 'wb') as model_file:
            model_file.write(r.content)
    
    model = joblib.load(model_filename)
    return model

def calculate_daily_calories(gender, weight, height, age, activity_factor):
    if gender.lower() == 'm':
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) + 5
    elif gender.lower() == 'f':
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) - 161
    else:
        raise ValueError("Invalid gender. Use 'm' or 'f'.")
    daily_calories = bmr * float(activity_factor)
    return daily_calories

def predict_calories(model, gender_encoded, age, height, weight, duration, pulse):
    user_input = np.array([[float(gender_encoded), float(age), float(height), float(weight), float(duration), float(pulse)]])
    return model.predict(user_input)

def display_food_suggestions(df_food, df_consumed, calories_difference):
    difference = df_consumed['Calories'].sum() - calories_difference
    suggestions = df_food[df_food['Calories'] < -difference][['Food', 'Calories', 'Sugar']]
    if not suggestions.empty:
        suggestions_no_index = suggestions.reset_index(drop=True)
        suggestions_no_index.index += 1
        return suggestions_no_index.sample(10)
    else:
        return pd.DataFrame()

def generate_top_foods(df, nutrient, column_name):
    top_foods = df.nlargest(10, nutrient)[['Food', nutrient, 'Sugar', 'Carbohydrate']]
    top_foods[nutrient] = top_foods[nutrient].apply(lambda x: f'{x:.2f}')
    top_foods['Sugar'] = top_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_foods['Carbohydrate'] = top_foods['Carbohydrate'].apply(lambda x: f'{x:.2f}')
    top_foods_no_index = top_foods.reset_index(drop=True)
    top_foods_no_index.index += 1
    st.write(top_foods_no_index)





