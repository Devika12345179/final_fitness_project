import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import time
import hashlib
import urllib.parse
import random
import warnings
warnings.filterwarnings('ignore')

# Background Image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://png.pngtree.com/background/20250103/original/pngtree-3d-rendered-gym-equipment-against-a-dark-backdrop-picture-image_11966449.jpg");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.write("## Personal Fitness Tracker")
st.write("Track your predicted calories burned based on your exercise parameters.")

# Motivational Quotes
quotes = [
    "Every step counts, keep moving forward!",
    "Fitness is not about being better than someone else. It’s about being better than you used to be.",
    "Push yourself, because no one else is going to do it for you.",
    "The only bad workout is the one that didn’t happen.",
    "Sweat is just fat crying!"
]
st.sidebar.write("💪 **Motivational Quote:**")
st.sidebar.write(np.random.choice(quotes))

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=170)
    bmi = round(weight / ((height / 100) ** 2), 2)  # Automatic BMI calculation
    
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    duration = st.sidebar.slider("Duration (min): ", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 180, 100)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    
    # Display BMI
    st.sidebar.write(f"Your BMI: **{bmi}**")
    
    data_model = {"Age": age, "BMI": bmi, "Duration": duration, "Heart_Rate": heart_rate, "Body_Temp": body_temp, "Gender_male": gender}
    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Ensure BMI is recalculated correctly
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Check if BMI column exists after calculation
assert 'BMI' in exercise_df.columns, "BMI column is missing in the exercise dataframe"

# Split data into training and testing
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)[0]

st.write("---")
st.header("Prediction: ")
st.write(f"{round(prediction, 2)} **kilocalories burned**")

# Save User Data for Progress Tracking
user_data = df.copy()
user_data["Calories_Burned"] = round(prediction, 2)
if not os.path.exists("user_history.csv"):
    user_data.to_csv("user_history.csv", index=False)
else:
    user_data.to_csv("user_history.csv", mode="a", header=False, index=False)

st.write("✅ Your data has been saved successfully!")

# Display User History
if os.path.exists("user_history.csv"):
    history = pd.read_csv("user_history.csv")
    st.write("### Your Previous Records")
    st.write(history.tail(5))

# Find similar results based on predicted calories
calorie_range = [prediction - 10, prediction + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write("### Similar Results")
st.write(similar_data.sample(5))


# Create a 2x3 grid layout for the plots (2 rows and 3 columns)
st.write("---")
st.header("Visual Insights")

# 2x3 grid of plots
col1, col2, col3 = st.columns(3)

with col1:
    # 1. Calories Burned Distribution
    fig, ax = plt.subplots()
    sns.histplot(exercise_df['Calories'], kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribution of Calories Burned")
    ax.set_xlabel("Calories")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 2. BMI Distribution
    fig, ax = plt.subplots()
    sns.histplot(exercise_df['BMI'], kde=True, ax=ax, color='orange')
    ax.set_title("Distribution of BMI")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 3. Heart Rate Distribution
    fig, ax = plt.subplots()
    sns.histplot(exercise_df['Heart_Rate'], kde=True, ax=ax, color='red')
    ax.set_title("Distribution of Heart Rate")
    ax.set_xlabel("Heart Rate")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with col2:
    # 4. Age Distribution
    fig, ax = plt.subplots()
    sns.histplot(exercise_df['Age'], kde=True, ax=ax, color='green')
    ax.set_title("Distribution of Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 5. Exercise Duration Distribution
    fig, ax = plt.subplots()
    sns.histplot(exercise_df['Duration'], kde=True, ax=ax, color='purple')
    ax.set_title("Distribution of Exercise Duration")
    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 6. Body Temperature Distribution
    fig, ax = plt.subplots()
    sns.histplot(exercise_df['Body_Temp'], kde=True, ax=ax, color='cyan')
    ax.set_title("Distribution of Body Temperature")
    ax.set_xlabel("Body Temperature (C)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


# Adding Google Search feature

st.write("---")
st.header("Search Google")
st.write("If you have any queries, you can search on Google directly from here!")

# User input for query
query = st.text_input("Enter your query:")

if query:
    # URL encode the query to make it valid for a search URL
    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
    st.write(f"Here is the [Google search link for your query](%s)" % search_url)
    st.write("Or you can simply click the link above to see the results!")
    

# --- Live Fitness Classes/Tutorials Section ---
st.write("## Live Fitness Classes/Tutorials")
st.write("Join live workout classes or view helpful fitness tutorials on YouTube!")

# YouTube Playlist or Live Stream Integration
st.write("### Recommended Fitness Classes")
st.markdown("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/6yfkX6Rf0mQ" 
    title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    """, unsafe_allow_html=True)

st.write("### More workout tutorials:")
st.write("1. [Morning Stretch Routine](https://www.youtube.com/watch?v=itJE4neqDJw)")
st.write("2. [High-Intensity Interval Training (HIIT)](https://www.youtube.com/watch?v=hOHhelsCHu4)")
st.write("3. [Yoga for Flexibility](https://www.youtube.com/watch?v=U-oPOj0W9Sc)")

# --- Gamification Features Section ---
st.write("## Gamification Features")
st.write("Earn points, unlock badges, and track your fitness journey with streaks!")

# Points System (Awarding Points based on Activity)
def calculate_points(duration, heart_rate):
    points = 0
    if duration >= 30:
        points += 10  # For 30 minutes or more
    if heart_rate > 140:
        points += 5  # Extra points for high heart rate (intensity)
    return points

# Check User Activity (input parameters)
exercise_duration = st.slider("Exercise Duration (min)", 0, 120, 30)
exercise_heart_rate = st.slider("Heart Rate", 60, 200, 100)

# Calculate Points
points_earned = calculate_points(exercise_duration, exercise_heart_rate)

# Display Points
st.write(f"💪 You earned **{points_earned}** points for this activity!")

# Streak Tracking (Store Streak in CSV)
if not os.path.exists("user_streak.csv"):
    streak_data = pd.DataFrame(columns=["User_ID", "Streak"])
    streak_data.to_csv("user_streak.csv", index=False)

# Load streak data to track user streaks
user_streak = pd.read_csv("user_streak.csv")

# Assume unique user is identified by a user ID (for simplicity, we use a random ID here)
user_id = random.randint(1000, 9999)

# Check if user has a streak in the system
if user_id in user_streak["User_ID"].values:
    current_streak = user_streak[user_streak["User_ID"] == user_id]["Streak"].values[0]
    current_streak += 1
    user_streak.loc[user_streak["User_ID"] == user_id, "Streak"] = current_streak
else:
    current_streak = 1
    new_data = pd.DataFrame({"User_ID": [user_id], "Streak": [current_streak]})
    user_streak = pd.concat([user_streak, new_data], ignore_index=True)

# Update the streak file with new data
user_streak.to_csv("user_streak.csv", index=False)

# Display Streak
st.write(f"🎉 You are on a **{current_streak}** day streak!")
if current_streak >= 7:
    st.write("🏆 Congratulations! You've earned the '7-Day Streak' badge!")

# Badges (Earn badges based on activity)
def check_badges(points, streak):
    badges = []
    if points >= 20:
        badges.append("🔥 'High Performer' Badge")
    if streak >= 7:
        badges.append("🏅 '7-Day Streak' Badge")
    return badges

# Check for any earned badges
earned_badges = check_badges(points_earned, current_streak)
if earned_badges:
    st.write("🎖️ Badges Earned: ")
    for badge in earned_badges:
        st.write(badge)

# Save progress for the user (e.g., points and streaks)
user_data = pd.DataFrame({
    "User_ID": [user_id],
    "Points": [points_earned],
    "Streak": [current_streak],
    "Badges": [', '.join(earned_badges) if earned_badges else 'None']
})
if not os.path.exists("user_progress.csv"):
    user_data.to_csv("user_progress.csv", index=False)
else:
    user_data.to_csv("user_progress.csv", mode="a", header=False, index=False)

st.write("✅ Your progress has been saved successfully!")

# --- Display User Progress History ---
if os.path.exists("user_progress.csv"):
    progress_history = pd.read_csv("user_progress.csv")
    st.write("### Your Progress History")
    st.write(progress_history.tail(5))  # Show the last 5 entries


# Feedback Section: Rating and Comments
st.write("---")
st.header("Feedback")
st.write("We would love to hear your thoughts on the accuracy of the predicted calories burned and the app's usability.")

# Add a rating system
rating = st.slider("How would you rate the prediction accuracy?", 1, 5, 3)
st.write(f"Your rating: {rating} out of 5")

