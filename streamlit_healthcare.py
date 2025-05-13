import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 

st.title("Data Analysis and Prediction")


gender = st.selectbox('gender',['Female','Male'])
age = st.number_input('age',min_value=0,max_value=100,value=20)
hypertension = st.selectbox('hypertension',[0,1])
heart_disease = st.selectbox('heart_disease',[0,1])
ever_married = st.selectbox('ever_married',['No','Yes'])
work_type = st.selectbox('work_type',['children','Govt_job','Never_worked','Private','Self-employed'])
residence_type = st.selectbox('residence_type',['Rural','Urban'])
avg_glucose_level = st.number_input('avg_glucose_level',min_value=0.0,max_value=300.0,value=100.0)
bmi = st.number_input('bmi',min_value=0.0,max_value=100.0,value=20.0)
smoking_status = st.selectbox('smoking_status',['formerly smoked','never smoked','smokes','Unknown'])

gender_binary = 1 if gender == 'Male' else 0
married = 1 if ever_married == 'Yes' else 0
worktype = {'children':0 , "Govt_job":1, "Never_worked":2, "Private":3, "Self-employed":4}
work_encoded = worktype[work_type]

res_type = 1 if residence_type == 'Urban' else 0
smoking = {'formerly smoked':0 , 'never smoked':1, 'smokes':2, 'Unknown':3}
smoking_encoded = smoking[smoking_status]

input_data = np.array([
    gender_binary, age, hypertension, heart_disease, married,
    work_encoded, res_type, avg_glucose_level, bmi,smoking_encoded
]).reshape(1, -1)

if st.button("Predict"):
    try:
        model = pickle.load(open(r'C:/Users/dell/Desktop/python/Machine learning/Project healthcare/rf_healthcare.pkl', 'rb'))
        prediction = model.predict(input_data)
        
        if prediction[0] == 1 :
            st.error("Ce patient a eu un AVC.")
        else:
            st.success("Ce patient n'a pas eu un AVC.")
    
    except Exception as e :
        st.error(f"Erreur lors de la prédiction : {e}")


if st.button("Analyse et Visualisation"):

    st.subheader("Chargement des données")
    try:
        df = pd.read_csv("healthcare-dataset-stroke-data.csv")
        st.write("Aperçu du dataset :")
        st.dataframe(df.head())

        st.subheader("Statistiques générales")
        st.write(df.describe())

        st.subheader("Distribution de l'âge")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Boxplot de l'âge selon le genre")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='gender', y='age', ax=ax2)
        st.pyplot(fig2)

        st.subheader("Taux d'AVC par type de travail")
        fig3, ax3 = plt.subplots()
        sns.countplot(data=df, x='work_type', hue='stroke', ax=ax3)
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        st.subheader("Corrélation entre les variables numériques")
        corr = df.corr(numeric_only=True)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Erreur lors du chargement ou de l'analyse des données : {e}")
        
