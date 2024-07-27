import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from functions import (
    load_dfs,
    load_model,
    calculate_daily_calories,
    predict_calories,
    display_food_suggestions,
    generate_top_foods
)

# Function to add meta tags for social media previews
def add_meta_tags():
    st.markdown(
        """
        <head>
            <meta property="og:title" content="Health Calculator">
            <meta property="og:description" content="A Streamlit app to calculate your daily caloric requirements and suggest healthy foods.">
            <meta property="og:image" content="URL_TO_YOUR_IMAGE">
            <meta property="og:url" content="URL_OF_YOUR_APP">
            <meta name="twitter:card" content="summary_large_image">
        </head>
        """,
        unsafe_allow_html=True,
    )

# Add meta tags to the app
add_meta_tags()

# Load CSS (check if exists to prevent file not found error)
try:
    with open("style.css") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("The style.css file was not found. Ensure it exists in the correct path.")

# Load data and model
try:
    df_food, df_recomm = load_dfs()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")

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

height = col1.text_input("Your height:", placeholder="e.g., 170")
weight = col2.text_input("Your weight:", placeholder="e.g., 80")
age = col3.text_input("Your age:", placeholder="e.g., 40")
gender = col4.text_input("Your gender (m/f):", placeholder="e.g., f").lower()
activity_factor = col5.text_input("Your physical activity level (1 low, 1.3 med, 1.5 high):", placeholder="e.g., 1.3")

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
selected_foods = st.multiselect("Select foods from the list:", options=df_food['Food'])
df_consumed = df_food[df_food['Food'].isin(selected_foods)].round(2)

if st.button("Submit"):
    st.subheader("Your today's result:")
    total_row = df_consumed.drop('Food', axis=1).sum(numeric_only=True).round(2)
    total_row['Food'] = 'TOTAL TODAY'
    df_recomm['Food'] = 'RECOMMENDED ' + df_recomm['Food']
    your_score_row = pd.Series(0, index=df_consumed.columns[1:], name=-1).round(2)
    your_score_row['Food'] = 'YOUR SCORE'   
    df_consumed = pd.concat([df_consumed, total_row.to_frame().T, df_recomm, your_score_row.to_frame().T])
    df_consumed.iloc[-2, df_consumed.columns.get_loc('Calories')] = recommended_calories
    df_consumed.iloc[-1, -27:] = (df_consumed.iloc[-3, -27:].values - df_consumed.iloc[-2, -27:].values).astype(float).round(2)
    df_consumed = df_consumed.reset_index(drop=True).drop(['Unnamed: 0'], axis=1)
    df_consumed.set_index('Food', inplace=True)
    st.write(df_consumed)
    
    # Diaplay the plot
    st.subheader("Check your nutrition score for today on the chart")

    # Extract the relevant row and ensure numeric
    row_to_display = df_consumed.iloc[-1, 1:].apply(pd.to_numeric, errors='coerce')  # Skip the 'Food' column and ensure numeric
    recommended_values = df_recomm.iloc[0, 1:].apply(pd.to_numeric, errors='coerce')  # Skip the 'Food' column and ensure numeric

    # Calculate the difference
    diff = row_to_display - recommended_values

    # Convert differences to a DataFrame for plotting
    diff_df = diff.reset_index()
    diff_df.columns = ['Nutrient', 'Difference']

    # Sort differences from the lowest to the highest
    diff_df = diff_df.sort_values(by='Difference')

    # Add a column for color based on the difference value
    diff_df['Color'] = diff_df['Difference'].apply(lambda x: 'lack' if x < 0 else 'plenty')

    # Create the bar plot using Plotly with red and green coloring
    fig = px.bar(diff_df, x='Difference', y='Nutrient', orientation='h',
             labels={'Difference': 'Difference from Recommended Values', 'Nutrient': 'Nutrient', 'Color': 'Score'},
             title='   Nutrients Difference from Recommended Intake',
             color='Color', color_discrete_map={'lack': 'red', 'plenty': 'green'})

    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig)

# Additional food suggestions
calories_difference = df_consumed['Calories'].sum() - recommended_calories
if calories_difference > 0:
    st.subheader(f"You have {calories_difference} kcal to store or burn today")
    st.subheader("Do you want to store or burn your calories?")
    duration = st.slider("Minutes of activity:", 1, 180)
    pulse = st.slider("Your pulse:, approx. 115-120 for light, 120-135 for fitness, 165-175 for aerobic, 165-175 for anaerobic, 175-185 max):", 70, 200)
    col1, col2 = st.columns(2)
    if col1.button("Store"):
        st.subheader("Congrats! You stored your calories!")
    if col2.button("Burn"):
        prediction = predict_calories(model, gender_encoded, age, height, weight, duration, pulse)
        st.subheader(f"Predicted Calories Burned {abs(prediction[0]):.2f} kcal 🏓")
else:
    calories_difference = abs(calories_difference)
    st.subheader(f"Well done! You have {calories_difference} more kcal to consume today. Do you want me to suggest you something nice?")
    col1, col2 = st.columns(2)
    if col1.button("Yes"):
        st.subheader("Here's a list of food suggestions 🥗")
        suggestions = display_food_suggestions(df_food, df_consumed, calories_difference)
        if suggestions.empty:
            st.write("No suitable suggestions found based on your criteria. Here's a list of all available foods:")
            st.dataframe(df_food[['Food', 'Calories', 'Sugar']])
        else:
            st.dataframe(suggestions)
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
st.subheader("Vitamins")
col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("Vitamin A"):
    st.write(generate_top_foods(df_food, 'Vitamin A', 'Vitamin A'))
if col2.button("Vitamin B6"):
    st.write(generate_top_foods(df_food, 'Vitamin B6', 'Vitamin B6'))
if col3.button("Vitamin B12"):
    st.write(generate_top_foods(df_food, 'Vitamin B12', 'Vitamin B12'))
if col4.button("Vitamin C"):
    st.write(generate_top_foods(df_food, 'Vitamin C', 'Vitamin C'))
if col5.button("Vitamin D"):
    st.write(generate_top_foods(df_food, 'Vitamin D', 'Vitamin D'))

col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("Vitamin K"):
    st.write(generate_top_foods(df_food, 'Vitamin K', 'Vitamin K'))
if col2.button("Selenium"):
    st.write(generate_top_foods(df_food, 'Selenium', 'Selenium'))
if col3.button("Thiamin"):
    st.write(generate_top_foods(df_food, 'Thiamin', 'Thiamin'))
if col4.button("Niacin"):
    st.write(generate_top_foods(df_food, 'Niacin', 'Niacin'))
if col5.button("Folate"):
    st.write(generate_top_foods(df_food, 'Folate', 'Folate'))

# Minerals section
st.subheader("Minerals")
col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("Fiber"):
    st.write(generate_top_foods(df_food, 'Fiber', 'Fiber'))
if col2.button("Protein"):
    st.write(generate_top_foods(df_food, 'Protein', 'Protein'))
if col3.button("Iron"):
    st.write(generate_top_foods(df_food, 'Iron', 'Iron'))
if col4.button("Phosphorus"):
    st.write(generate_top_foods(df_food, 'Phosphorus', 'Phosphorus'))
if col5.button("Polyunsaturated (good) Fat"):
    st.write(generate_top_foods(df_food, 'Polyunsaturated (good) Fat', 'Polyunsaturated (good) Fat'))

col1, col2, col3, col4, col5, col6 = st.columns(6)
if col1.button("Copper"):
    st.write(generate_top_foods(df_food, 'Copper', 'Copper'))
if col2.button("Choline"):
    st.write(generate_top_foods(df_food, 'Choline', 'Choline'))
if col3.button("Calcium"):
    st.write(generate_top_foods(df_food, 'Calcium', 'Calcium'))
if col4.button("Magnesium"):
    st.write(generate_top_foods(df_food, 'Magnesium', 'Magnesium'))
if col5.button("Potassium"):
    st.write(generate_top_foods(df_food, 'Potassium', 'Potassium'))
if col6.button("Zinc"):
    st.write(generate_top_foods(df_food, 'Zinc', 'Zinc'))



