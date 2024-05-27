import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# Stress Detection
Stress indicator using Machine Learning!
""")

image = Image.open('dep.png')
st.image(image, caption='ML', use_column_width=True)

df = pd.read_csv('SaYoPillow.csv')

st.subheader('Data Information: ')
st.dataframe(df)

st.write(df.describe())
chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

def get_user_input():
    sr = st.sidebar.slider('Snoring Range', 40.00, 100.00, 69.696)
    rr = st.sidebar.slider('Respiration Rate', 15.000, 30.000, 25.456)
    t = st.sidebar.slider('Body Temperature', 85.000, 100.000, 96.648)
    lm = st.sidebar.slider('Limb Movement Rate', 5.000, 20.000, 9.658)
    bo = st.sidebar.slider('Blood Oxygen Levels', 80.000, 100.000, 91.392)
    rem = st.sidebar.slider('Eye Movement', 50.00, 110.00, 104.64)
    hs = st.sidebar.slider('Hours of Sleep', 0.000, 10.000, 4.688)
    hr = st.sidebar.slider('Heart Rate', 50.00, 90.00, 84.76)

    user_data = {'Snoring Range': sr,
                'Respiration Rate':rr,
                'Body Temperature':t,
                'Limb Movement Rate':lm,
                'Blood Oxygen Levels':bo,
                'Eye Movement':rem,
                'Hours of Sleep':hs,
                'Heart Rate':hr,
                }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()
st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')


prediction = RandomForestClassifier.predict(user_input)
st.subheader('Classification: ')
st.write(prediction)

