import streamlit as st
from functions import (
    load_dfs,
    load_model,
    calculate_daily_calories,
    predict_calories,
    display_food_suggestions,
    generate_top_foods
)
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

# Load CSS
st.markdown(
    """
    <style>
        %s
    </style>
    """ % open("style.css").read(),
    unsafe_allow_html=True,
)

# Load data and model
df_food, df_recomm = load_dfs()
model = load_model()

# App Header
st.markdown("<h1 style='text-align: center;'>Health Calculator 🍎 </h1>", unsafe_allow_html=True)
st.subheader('Philosophy behind the App')
st.write("""
    This app is not for weight loss. It was designed to raise awareness of everyday consumption, which has proven long-term consequences. 
    The food data frame was sourced from data.gov, and the recommendation data frame is based on US National Library of Medicine publications. 
    Your data is not being saved by the app because it prioritizes maintaining a daily 5-minute balance over creating complex graphs.
    Enjoy and stay healthy!
""")

# User's parameters input
col1 = st.columns(1)[0]
col2, col3 = st.columns(2)
col4, col5 = st.columns(2)

height = col1.text_input("Your height:")
weight = col2.text_input("Your weight:")
age = col3.text_input("Your age:")
gender = col4.text_input("Your gender (m/f):").lower()
activity_factor = col5.text_input("Your physical activity level (1 low, 1.3 med, 1.5 high):")

# Gender encoding
label_encoder = {'m': 0, 'f': 1}
gender_encoded = label_encoder.get(gender, None)

recommended_calories = None

# Calculate recommended calories
if gender_encoded is not None:
    try:
        height = float(height)
        weight = float(weight)
        age = int(age)
        activity_factor = float(activity_factor)
        recommended_calories = calculate_daily_calories(gender, weight, height, age, activity_factor)
        st.subheader(f"Your recommended daily calories: {recommended_calories}")
    except ValueError:
        st.warning("Please enter valid numbers for height, weight, age, and activity factor.")
else:
    st.warning("Invalid gender. Please enter 'm' or 'f'.")

# User's food input
st.subheader("What did you eat today?")
selected_foods = st.multiselect("", options=df_food['Food'])
df_consumed = df_food[df_food['Food'].isin(selected_foods)].round(2)


if st.button("Submit"):
  st.write("Your today's result:") 
  st.write(df_consumed)
    
# Remove non-numeric columns and rows
row_to_display = df_consumed.iloc[-1, 1:].apply(pd.to_numeric, errors='coerce')  # Skip the 'Food' column and ensure numeric
recommended_values = df_recomm.iloc[0, 1:].apply(pd.to_numeric, errors='coerce')  # Skip the 'Food' column and ensure numeric
diff = row_to_display - recommended_values
    
# Sort differences from the biggest lack to the lowest
diff = diff.sort_values()

fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(x=diff.values, y=diff.index, palette=['red' if x < 0 else 'green' for x in diff])
    
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Difference from Recommended Values', fontsize=14)
ax.set_ylabel('Nutrient', fontsize=14)
ax.set_title('Nutrients Difference from Recommended Intake', fontsize=16)
    
# Increase the size of tick labels
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
    
plt.xticks(rotation=90)
st.pyplot(fig)

# Additional food suggestions
if recommended_calories is not None:
    calories_difference = df_consumed['Calories'].sum() - recommended_calories
    if calories_difference > 0:
        st.subheader(f"You have {calories_difference} kcal to store or burn today")
        st.subheader("Do you want to store or burn your calories?")
        duration = st.slider("Minutes of activity:", 1, 180)
        pulse = st.slider("Your pulse:", 70, 200)
        col1, col2 = st.columns(2)
        if col1.button("Store"):
            st.subheader("Congrats! You stored your calories!")
        if col2.button("Burn"):
            prediction = predict_calories(model, gender_encoded, age, height, weight, duration, pulse)
            st.subheader(f"Predicted Calories Burned {abs(prediction[0]):.2f} kcal 🏓")
            st.write("You can trust the prediction. The R2 of the model is 0.95!")
    else:
        st.subheader(f"Well done! You have {-calories_difference} more kcal to consume today. Do you want me to suggest you something nice?")
        col1, col2 = st.columns(2)
        if col1.button("Yes"):
            display_food_suggestions(df_food, df_consumed, calories_difference)
        if col2.button("No"):
            st.subheader("Good luck with your choices!")

# Vitamin and Mineral buttons
st.subheader("Look what you can eat to fulfill nutrition gaps")

vitamins = {
    "Vitamin A": "Vitamin A",
    "Vitamin B6": "Vitamin B6",
    "Vitamin B12": "Vitamin B12",
    "Vitamin C": "Vitamin C",
    "Vitamin D": "Vitamin D",
    "Vitamin K": "Vitamin K",
    "Selenium": "Selenium",
    "Thiamin": "Thiamin",
    "Niacin": "Niacin",
    "Folate": "Folate"
}

minerals = {
    "Fiber": "Fiber",
    "Protein": "Protein",
    "Copper": "Copper",
    "Choline": "Choline",
    "Calcium": "Calcium",
    "Iron": "Iron",
    "Magnesium": "Magnesium",
    "Zinc": "Zinc",
    "Potassium": "Potassium",
    "Phosphorus": "Phosphorus",
    "Polyunsaturated (good) Fat": "Polyunsaturated (good) Fat"
}

# Vitamins section
st.subheader("")
st.subheader("Vitamins")
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Vitamin A"):
    generate_top_foods(df_food, 'Vitamin A', 'Vitamin A')
if col2.button("Vitamin B6"):
    generate_top_foods(df_food, 'Vitamin B6', 'Vitamin B6')
if col3.button("Vitamin B12"):
    generate_top_foods(df_food, 'Vitamin B12', 'Vitamin B12')
if col4.button("Vitamin C"):
    generate_top_foods(df_food, 'Vitamin C', 'Vitamin C')
if col5.button("Vitamin D"):
    generate_top_foods(df_food, 'Vitamin D', 'Vitamin D')
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Vitamin K"):
    generate_top_foods(df_food, 'Vitamin K', 'Vitamin K')
if col2.button("Selenium"):
    generate_top_foods(df_food, 'Selenium', 'Selenium')
if col3.button("Thiamin"):
    generate_top_foods(df_food, 'Thiamin', 'Thiamin')
if col4.button("Niacin"):
    generate_top_foods(df_food, 'Niacin', 'Niacin')
if col5.button("Folate"):
    generate_top_foods(df_food, 'Folate', 'Folate')

# Minerals section
st.subheader("")
st.subheader("Minerals")
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Fiber"):
    generate_top_foods(df_food, 'Fiber', 'Fiber')
if col2.button("Protein"):
    generate_top_foods(df_food, 'Protein', 'Protein')
if col3.button("Copper"):
    generate_top_foods(df_food, 'Copper', 'Copper')
if col4.button("Choline"):
    generate_top_foods(df_food, 'Choline', 'Choline')
if col5.button("Calcium"):
    generate_top_foods(df_food, 'Calcium', 'Calcium')
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Iron"):
    generate_top_foods(df_food, 'Iron', 'Iron')
if col2.button("Magnesium"):
    generate_top_foods(df_food, 'Magnesium', 'Magnesium')
if col3.button("Zinc"):
    generate_top_foods(df_food, 'Zinc', 'Zinc')
if col4.button("Potassium"):
    generate_top_foods(df_food, 'Potassium', 'Potassium')
if col5.button("Phosphorus"):
    generate_top_foods(df_food, 'Phosphorus', 'Phosphorus')
if col1.button("Polyunsaturated (good) Fat"):
    generate_top_foods(df_food, 'Polyunsaturated (good) Fat', 'Polyunsaturated (good) Fat')
