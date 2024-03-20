import streamlit as st 
import streamlit.components.v1 as com
import pandas as pd
import numpy as np
import seaborn as sns
import joblib

from matplotlib import pyplot as plt
figure=plt.figure()

import warnings
warnings.filterwarnings("ignore")

st.markdown(
    """
    <style>
        %s
    </style>
    """ % open("style.css").read(),
    unsafe_allow_html=True,
) 
def load_dfs():
    key1 = 'df_food'
    key2 = 'recommendations'
    uploaded_file1 = st.file_uploader("Upload your food file", type=['csv'], key=key1)
    uploaded_file2 = st.file_uploader("Upload your recommendations file", type=['csv'], key=key2)
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    if uploaded_file1 is not None:
        df1 = pd.read_csv(uploaded_file1)
    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
    return df1, df2

# CALCULATIONS FUNCTIONS
def calculate_daily_calories(gender, weight, height, age, activity_factor):
    if gender.lower() == 'male':
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) + 5
    elif gender.lower() == 'female':
        bmr = (10 * float(weight)) + (6.25 * float(height)) - (5 * float(age)) - 161
    else:
        raise ValueError("Invalid gender. Use 'male' or 'female'.")
    daily_calories = bmr * float(activity_factor)
    return daily_calories

# USER'S INPUT
def get_user_input(df_food, df_recomm):
    global df_consumed
    st.subheader("")
    st.subheader("What did you eat today?")
    selected_foods = st.multiselect("", options=df_food['Food'])
    df_consumed = df_food[df_food['Food'].isin(selected_foods)]
    total_row = df_consumed.drop('Food', axis=1).sum(numeric_only=True)
    total_row['Food'] = 'TOTAL TODAY'

    df_recomm['Food'] = 'RECOMMENDED ' + df_recomm['Food']
    recommended_calories = calculate_daily_calories(gender, weight, height, age, activity_factor)
    
    your_score_row = pd.Series(0, index=df_consumed.columns, name=-1)
    your_score_row['Food'] = 'YOUR SCORE'
    
    df_consumed = pd.concat([df_consumed, total_row.to_frame().T, df_recomm, your_score_row.to_frame().T]) 
    df_consumed.iloc[-2, df_consumed.columns.get_loc('Calories')] = recommended_calories

    df_consumed.iloc[-1, -27:] = (df_consumed.iloc[-3, -27:].values - df_consumed.iloc[-2, -27:].values).astype(float).round(2)

    df_consumed = df_consumed.reset_index(drop=True).drop(['Unnamed: 0'], axis=1)
    df_consumed.set_index('Food', inplace=True)
    
    if st.button("Submit"):
        st.write("Your today's result:") 
        st.write(df_consumed)  
        row_to_display = df_consumed.iloc[-1, :].astype(float)  # Ensure the values are numeric
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(row_to_display.to_frame(), annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('Nutrients')
        ax.set_xlabel('')
        ax.set_ylabel('')
        st.pyplot(fig)

# ML FUNCTIONS 
def predict_calories(gender_encoded, age, height, weight, duration, pulse):
    model = joblib.load('https://github.com/OlgaGnezdilova/Health-Calculator/blob/main/linear_regression_model.pkl')
    #prediction = None
    user_input = np.array([[float(gender_encoded), float(age), float(height), float(weight), float(duration), float(pulse)]])
    return model.predict(user_input)

def btn_click_store():
    st.subheader("Congrats! You stored your calories!")

def btn_click_burn():
    prediction = predict_calories(gender_encoded, age, height, weight, duration, pulse)
    prediction = abs(prediction)
    st.subheader(f"Predicted Calories Burned {prediction[0]:.2f} kcal 🏓" )  
    st.write("You can trust the prediction. The R2 of the model is 0.95!")

def display_food_suggestions(df_food, calories_difference):
    difference = df_consumed['Calories'].sum() - recommended_calories
    display_food_suggestions(df_consumed, difference)


# VITAMIN FUNCTIONS
def btn_click_vitA():
    st.subheader("Here is Top 10 of Vitamin A-rich food: ")
    top_vitA_foods = df_food.nlargest(10, 'Vitamin A')[['Food', 'Vitamin A','Sugar']]
    top_vitA_foods['Vitamin A'] = top_vitA_foods['Vitamin A'].apply(lambda x: f'{x:.2f}')
    top_vitA_foods['Sugar'] = top_vitA_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_vitA_foods_no_index = top_vitA_foods.reset_index(drop=True)  
    top_vitA_foods_no_index.index += 1
    st.write(top_vitA_foods_no_index)

def btn_click_vitB6():
    st.subheader("Here is Top 10 of Vitamin B6-rich food: ")
    top_vitB6_foods = df_food.nlargest(10, 'Vitamin B6')[['Food', 'Vitamin B6','Sugar']]
    top_vitB6_foods['Vitamin B6'] = top_vitB6_foods['Vitamin B6'].apply(lambda x: f'{x:.2f}')
    top_vitB6_foods['Sugar'] = top_vitB6_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_vitB6_foods_no_index = top_vitB6_foods.reset_index(drop=True)  
    top_vitB6_foods_no_index.index += 1
    st.write(top_vitB6_foods_no_index)
   
def btn_click_vitB12():
    st.subheader("Here is Top 10 of Vitamin B12-rich food: ")
    top_vitB12_foods = df_food.nlargest(10, 'Vitamin B12')[['Food', 'Vitamin B12','Sugar']]
    top_vitB12_foods['Vitamin B12'] = top_vitB12_foods['Vitamin B12'].apply(lambda x: f'{x:.2f}')
    top_vitB12_foods['Sugar'] = top_vitB12_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_vitB12_foods_no_index = top_vitB12_foods.reset_index(drop=True)  
    top_vitB12_foods_no_index.index += 1
    st.write(top_vitB12_foods_no_index)  

def btn_click_vitC():
    st.subheader("Here is Top 10 of Vitamin C-rich food: ")
    top_vitC_foods = df_food.nlargest(10, 'Vitamin C')[['Food', 'Vitamin C','Sugar']]
    top_vitC_foods['Vitamin C'] = top_vitC_foods['Vitamin C'].apply(lambda x: f'{x:.2f}')
    top_vitC_foods['Sugar'] = top_vitC_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_vitC_foods_no_index = top_vitC_foods.reset_index(drop=True)  
    top_vitC_foods_no_index.index += 1
    st.write(top_vitC_foods_no_index)    

def btn_click_vitD():
    st.subheader("Here is Top 10 of Vitamin D-rich food: ")
    top_vitD_foods = df_food.nlargest(10, 'Vitamin D')[['Food', 'Vitamin D','Sugar']]
    top_vitD_foods['Vitamin D'] = top_vitD_foods['Vitamin D'].apply(lambda x: f'{x:.2f}')
    top_vitD_foods['Sugar'] = top_vitD_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_vitD_foods_no_index = top_vitD_foods.reset_index(drop=True)  
    top_vitD_foods_no_index.index += 1
    st.write(top_vitD_foods_no_index)
 
def btn_click_vitK():
    st.subheader("Here is Top 10 of Vitamin K-rich food: ")
    top_vitK_foods = df_food.nlargest(10, 'Vitamin K')[['Food', 'Vitamin K','Sugar']]
    top_vitK_foods['Vitamin K'] = top_vitK_foods['Vitamin K'].apply(lambda x: f'{x:.2f}')
    top_vitK_foods['Sugar'] = top_vitK_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_vitK_foods_no_index = top_vitK_foods.reset_index(drop=True)  
    top_vitK_foods_no_index.index += 1
    st.write(top_vitK_foods_no_index)
   
def btn_click_selenium():
    st.subheader("Here is Top 10 of Selenium-rich food: ")
    top_selenium_foods = df_food.nlargest(10, 'Selenium')[['Food', 'Selenium','Sugar']]
    top_selenium_foods['Selenium'] = top_selenium_foods['Selenium'].apply(lambda x: f'{x:.2f}')
    top_selenium_foods['Sugar'] = top_selenium_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_selenium_foods_no_index = top_selenium_foods.reset_index(drop=True)  
    top_selenium_foods_no_index.index += 1
    st.write(top_selenium_foods_no_index)

def btn_click_thiamin():
    st.subheader("Here is Top 10 of Thiamin-rich food: ")
    top_thiamin_foods = df_food.nlargest(10, 'Thiamin')[['Food', 'Thiamin', 'Sugar']]
    top_thiamin_foods['Thiamin'] = top_thiamin_foods['Thiamin'].apply(lambda x: f'{x:.2f}')
    top_thiamin_foods['Sugar'] = top_thiamin_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_thiamin_foods_no_index = top_thiamin_foods.reset_index(drop=True)  
    top_thiamin_foods_no_index.index += 1
    st.write(top_thiamin_foods_no_index)
  
def btn_click_niacin():
    st.subheader("Here is Top 10 of Niacin-rich food: ")
    top_niacin_foods = df_food.nlargest(10, 'Niacin')[['Food', 'Niacin','Sugar']]
    top_niacin_foods['Niacin'] = top_niacin_foods['Niacin'].apply(lambda x: f'{x:.2f}')
    top_niacin_foods['Sugar'] = top_niacin_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_niacin_foods_no_index = top_niacin_foods.reset_index(drop=True)  
    top_niacin_foods_no_index.index += 1
    st.write(top_niacin_foods_no_index)
    
def btn_click_folate():
    st.subheader("Here is Top 10 of Folate-rich food: ")
    top_folate_foods = df_food.nlargest(10, 'Food Folate')[['Food', 'Food Folate','Sugar']]
    top_folate_foods['Food Folate'] = top_folate_foods['Food Folate'].apply(lambda x: f'{x:.2f}')
    top_folate_foods['Sugar'] = top_folate_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_folate_foods_no_index = top_folate_foods.reset_index(drop=True)  
    top_folate_foods_no_index.index += 1
    st.write(top_folate_foods_no_index)

# NUTRIENT FUNCTIONS 
def btn_click_choline():
    st.subheader("Here is Top 10 of Choline-rich food: ")
    top_choline_foods = df_food.nlargest(10, 'Choline')[['Food', 'Choline', 'Sugar']]
    top_choline_foods['Choline'] = top_choline_foods['Choline'].apply(lambda x: f'{x:.2f}')
    top_choline_foods['Sugar'] = top_choline_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_choline_foods_no_index = top_choline_foods.reset_index(drop=True)  
    top_choline_foods_no_index.index += 1
    st.write(top_choline_foods_no_index)

def btn_click_fiber():
    st.subheader("Here is Top 10 of Fiber-rich food: ")
    top_fiber_foods = df_food.nlargest(10, 'Fiber')[['Food', 'Fiber','Sugar']]
    top_fiber_foods['Fiber'] = top_fiber_foods['Fiber'].apply(lambda x: f'{x:.2f}')
    top_fiber_foods['Sugar'] = top_fiber_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_fiber_foods_no_index = top_fiber_foods.reset_index(drop=True)  
    top_fiber_foods_no_index.index += 1
    st.write(top_fiber_foods_no_index)

def btn_click_protein():
    st.subheader("Here is Top 10 of Protein-rich food: ")
    top_protein_foods = df_food.nlargest(10, 'Fiber')[['Food', 'Protein','Sugar']]
    top_protein_foods['Protein'] = top_protein_foods['Protein'].apply(lambda x: f'{x:.2f}')
    top_protein_foods['Sugar'] = top_protein_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_protein_foods_no_index = top_protein_foods.reset_index(drop=True)  
    top_protein_foods_no_index.index += 1
    st.write(top_protein_foods_no_index)

def btn_click_polyun():
    st.subheader("Here is Top 10 of Polyunsaturated (good) Fat-rich food: ")
    top_polyun_foods = df_food.nlargest(10, 'Polyunsaturated (good) Fat')[['Food', 'Polyunsaturated (good) Fat', 'Sugar']]
    top_polyun_foods['Polyunsaturated (good) Fat'] = top_polyun_foods['Polyunsaturated (good) Fat'].apply(lambda x: f'{x:.2f}')
    top_polyun_foods['Sugar'] = top_polyun_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_polyun_foods_no_index = top_polyun_foods.reset_index(drop=True)  
    top_polyun_foods_no_index.index += 1
    st.write(top_polyun_foods_no_index)

def btn_click_calcium():
    st.subheader("Here is Top 10 of Calcium-rich food: ")
    top_calcium_foods = df_food.nlargest(10, 'Calcium')[['Food', 'Calcium', 'Sugar']]
    top_calcium_foods['Calcium'] = top_calcium_foods['Calcium'].apply(lambda x: f'{x:.2f}')
    top_calcium_foods['Sugar'] = top_calcium_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_calcium_foods_no_index = top_calcium_foods.reset_index(drop=True)  
    top_calcium_foods_no_index.index += 1
    st.write(top_calcium_foods_no_index)

def btn_click_iron():
    st.subheader("Here is Top 10 of Iron-rich food: ")
    top_iron_foods = df_food.nlargest(10, 'Iron')[['Food', 'Iron', 'Sugar']]
    top_iron_foods['Iron'] = top_iron_foods['Iron'].apply(lambda x: f'{x:.2f}')
    top_iron_foods['Sugar'] = top_iron_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_iron_foods_no_index = top_iron_foods.reset_index(drop=True)  
    top_iron_foods_no_index.index += 1
    st.write(top_iron_foods_no_index)

def btn_click_magnesium():
    st.subheader("Here is Top 10 of Magnesium-rich food: ")
    top_magnesium_foods = df_food.nlargest(10, 'Magnesium')[['Food', 'Magnesium', 'Sugar']]
    top_magnesium_foods['Magnesium'] = top_magnesium_foods['Magnesium'].apply(lambda x: f'{x:.2f}')
    top_magnesium_foods['Sugar'] = top_magnesium_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_magnesium_foods_no_index = top_magnesium_foods.reset_index(drop=True)  
    top_magnesium_foods_no_index.index += 1
    st.write(top_magnesium_foods_no_index)

def btn_click_phos():
    st.subheader("Here is Top 10 of Phosphorus-rich food: ")
    top_phosphorus_foods = df_food.nlargest(10, 'Phosphorus')[['Food', 'Phosphorus', 'Sugar']]
    top_phosphorus_foods['Phosphorus'] = top_phosphorus_foods['Phosphorus'].apply(lambda x: f'{x:.2f}')
    top_phosphorus_foods['Sugar'] = top_phosphorus_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_phosphorus_foods_no_index = top_phosphorus_foods.reset_index(drop=True)  
    top_phosphorus_foods_no_index.index += 1
    st.write(top_phosphorus_foods_no_index)

def btn_click_potassium():
    st.subheader("Here is Top 10 of Potassium-rich food: ")
    top_potassium_foods = df_food.nlargest(10, 'Potassium')[['Food', 'Potassium', 'Sugar']]
    top_potassium_foods['Potassium'] = top_potassium_foods ['Potassium'].apply(lambda x: f'{x:.2f}')
    top_potassium_foods ['Sugar'] = top_potassium_foods ['Sugar'].apply(lambda x: f'{x:.2f}')
    top_potassium_foods_no_index = top_potassium_foods.reset_index(drop=True)  
    top_potassium_foods_no_index.index += 1
    st.write(top_potassium_foods)

def btn_click_zinc():
    st.subheader("Here is Top 10 of Zinc-rich food: ")
    top_zinc_foods = df_food.nlargest(10, 'Zinc')[['Food', 'Zinc', 'Sugar']]
    top_zinc_foods['Zinc'] = top_zinc_foods['Zinc'].apply(lambda x: f'{x:.2f}')
    top_zinc_foods['Sugar'] = top_zinc_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_zinc_foods_no_index = top_zinc_foods.reset_index(drop=True)  
    top_zinc_foods_no_index.index += 1
    st.write(top_zinc_foods_no_index)

def btn_click_copper():
    st.subheader("Here is Top 10 of Copper-rich food: ")
    top_copper_foods = df_food.nlargest(10, 'Copper')[['Food', 'Copper', 'Sugar']]
    top_copper_foods['Copper'] = top_copper_foods['Copper'].apply(lambda x: f'{x:.2f}')
    top_copper_foods['Sugar'] = top_copper_foods['Sugar'].apply(lambda x: f'{x:.2f}')
    top_copper_foods_no_index = top_copper_foods.reset_index(drop=True)  
    top_copper_foods_no_index.index += 1
    st.write(top_copper_foods_no_index)




# BODY OF THE CODE
st.markdown("<h1 style='text-align: center;'>Health Calculator 🍎 </h1>", unsafe_allow_html=True)
st.subheader("")
st.subheader('Philosophy behind the App')
st.write("This app is not for weight loss. It was designed to raise awareness of everyday consumption, which has proven long-term consequences. The food data frame was sourced from data.gov, and the recommendation data frame is based on US National Library of Medicine publications. Your data is not being saved by the app because it prioritizes maintaining a daily 5-minute balance over creating complex graphs.")
st.write("Enjoy and stay healthy!")
st.write("")

# Calculating recommended calories
st.subheader("Please, input your parameters")
col1,col2, col3=st.columns(3)
col4,col5=st.columns(2)
gender = col1.text_input("Your gender (male/female):").lower()
height = col2.text_input("Your height:")
weight = col3.text_input("Your weight:")
age = col4.text_input("Your age:")
activity_factor = col5.text_input("Your physical activity level (1 low, 1.3 med, 1.5 high):")
label_encoder = {'male': 0, 'female': 1}
gender_encoded = label_encoder.get(gender, None)
if gender_encoded is not None:
    recommended_calories = calculate_daily_calories(gender, weight, height, age, activity_factor)
    st.subheader('')
    st.subheader(f"Your recommended daily calories: {recommended_calories}")
else:
    st.warning("Invalid gender. Please enter 'male' or 'female'.")

# UPLOADING DFs 
st.sidebar.header("Upload your data frames")
with st.sidebar: 
    df_food, df_recomm = load_dfs()

# USER'S INPUT 🍉
get_user_input(df_food, df_recomm)

# OTHER FOOD SUGGESTIONS
calories_difference = df_consumed.iloc[-1, 0] 
if calories_difference > 0:
    st.subheader(f"You have {calories_difference} kcal to store or burn today")
    st.subheader("Do you want to store or burn your calories?")
    col1, col2 = st.columns(2)
    duration = st.slider("Minutes of activity:", 1, 180)
    pulse = st.slider("Your pulse (approx. 115-120 for light, 120-135 for fitness, 165-175 for aerobic, 165-175 for anaerobic, 175-185 max):", 70, 200)
    if col1.button("Store"):
        btn_click_store()
    if col2.button("Burn"):
        btn_click_burn()
else:
    st.subheader(f"Well done! You have {-calories_difference} more kcal to consume today. Do you want me to suggest you something nice?")
    col1, col2 = st.columns(2)
    btn1=col1.button("Yes")
    btn2=col2.button("No")
    st.write("", "", "", text_align="center")
    if btn1:
        df_difference = df_food[df_food['Calories'] < -calories_difference][['Food', 'Calories', 'Sugar']]
        if not df_difference.empty:
            st.subheader("Here's a list of food suggestions 🥗")
            df_difference_no_index = df_difference.copy().reset_index(drop=True)  # Reset index without mutating original DataFrame
            df_difference_no_index.iloc[:, -1] = df_difference_no_index.iloc[:, -1].apply(lambda x: f'{x:.2f}')
            df_difference_no_index.iloc[:, -2] = df_difference_no_index.iloc[:, -2].apply(lambda x: f'{x:.2f}')
            df_difference_no_index.index += 1
            st.write(df_difference_no_index.sample(10))
        else:
            st.write("No suggestions available. You're already meeting or exceeding your daily calorie goal!")
    if btn2:
        st.subheader("Good luck with your choices!")


# RICHEST FOOD BUTTONS
st.subheader("Look what you can eat to fulfill nutrition gaps")

# VITAMIN BUTTONS
st.subheader("")
st.subheader("Vitamins")
col1, col2, col3, col4, col5 = st.columns(5)
col1, col2, col3, col4, col5, col6 = st.columns(6)
if col1.button("Vitamin A"):
    btn_click_vitA()
if col2.button("Vitamin B6"):
    btn_click_vitB6()
if col3.button("Vitamin C"):
    btn_click_vitC()
if col4.button("Vitamin D"):
    btn_click_vitD()
if col5.button("Vitamin K"):
    btn_click_vitK()
if col1.button("Selenium"):
    btn_click_selenium()
if col2.button("Thiamin"):
    btn_click_thiamin()
if col3.button("Niacin"):
    btn_click_niacin()
if col4.button("Folate"):
    btn_click_folate()
if col5.button("Vitamin B12"):
    btn_click_vitB12()

# NUTRIENTS BUTTONS
st.subheader("")
st.subheader("Minerals")
col1, col2, col3, col4, col5 = st.columns(5)
st.markdown(" ", unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)
if col1.button("Fiber"):
    btn_click_fiber()
if col2.button("Protein"):
    btn_click_protein()
if col3.button("Copper"):
    btn_click_copper()
if col4.button("Choline"):
    btn_click_choline()
if col5.button("Calcium"):
    btn_click_calcium()
if col1.button("Iron"):
    btn_click_iron()
if col2.button("Magnesium"):
    btn_click_magnesium()
if col3.button("Zinc"):
    btn_click_zinc()
if col4.button("Potassium"):
    btn_click_potassium()
if col5.button("Phosphorus"):
    btn_click_phos()
if col6.button("Polyunsaturated (good) Fat"):
    btn_click_polyun()