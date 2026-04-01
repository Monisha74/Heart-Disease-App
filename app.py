import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Heart Disease AI", layout="wide")

# Load dataset
data = pd.read_csv("heart.csv")

# Load ML model
model = pickle.load(open("model.pkl","rb"))

# Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["Prediction","Dataset","Analytics","Admin"],
    icons=["heart","table","bar-chart","person"],
    orientation="horizontal"
)

# -------------------- Prediction Page --------------------

if selected == "Prediction":

    st.markdown("""
    <h1 style='text-align:center;color:#e63946'>
    ❤️ Heart Disease Risk Prediction
    </h1>
    """, unsafe_allow_html=True)

    st.write("Enter patient details below")

    name = st.text_input("Patient Name")

    col1,col2,col3 = st.columns(3)

    with col1:
        age = st.number_input("Age",20,80)

    with col2:
        sex = st.selectbox("Sex",[0,1],format_func=lambda x:"Female" if x==0 else "Male")

    with col3:
        cp = st.selectbox("Chest Pain Type",[0,1,2,3])

    col4,col5,col6 = st.columns(3)

    with col4:
        trtbps = st.slider("Resting Blood Pressure",90,200)

    with col5:
        chol = st.slider("Cholesterol",120,400)

    with col6:
        thalachh = st.slider("Max Heart Rate",70,210)

    col7,col8,col9 = st.columns(3)

    with col7:
        fbs = st.selectbox("Fasting Blood Sugar",[0,1])

    with col8:
        restecg = st.selectbox("ECG Result",[0,1,2])

    with col9:
        exng = st.selectbox("Exercise Angina",[0,1])

    col10,col11,col12 = st.columns(3)

    with col10:
        oldpeak = st.slider("Oldpeak",0.0,6.0)

    with col11:
        slp = st.selectbox("Slope",[0,1,2])

    with col12:
        caa = st.selectbox("Major Vessels",[0,1,2,3,4])

    thall = st.selectbox("Thalassemia",[0,1,2,3])

    input_data = np.array([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall]])

    if st.button("🔍 Predict Heart Disease"):

        # Validation
        if name.strip() == "":
            st.warning("⚠ Please enter patient name before prediction.")

        else:

            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            conn = sqlite3.connect("hospital.db")
            cursor = conn.cursor()

            if prediction == 0:

                st.error("⚠ High Risk of Heart Disease")

                cursor.execute("""
                INSERT INTO high_risk(name,age,bp,chol)
                VALUES (?,?,?,?)
                """,(name,age,trtbps,chol))

            else:

                st.success("✅ Low Risk of Heart Disease")

                cursor.execute("""
                INSERT INTO low_risk(name,age,bp,chol)
                VALUES (?,?,?,?)
                """,(name,age,trtbps,chol))

            conn.commit()
            conn.close()

            st.progress(int(probability*100))
            st.write("Risk Probability:",round(probability*100,2),"%")

# -------------------- Dataset Page --------------------

elif selected == "Dataset":

    st.title("📋 Heart Disease Dataset")

    st.dataframe(data)

    st.write("Dataset Shape:",data.shape)

# -------------------- Analytics Page --------------------

elif selected == "Analytics":

    st.title("📊 Heart Disease Analytics")

    col1,col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            data,
            x="age",
            color="output",
            title="Age Distribution vs Heart Disease"
        )
        st.plotly_chart(fig1)

    with col2:
        fig2 = px.scatter(
            data,
            x="chol",
            y="thalachh",
            color="output",
            title="Cholesterol vs Heart Rate"
        )
        st.plotly_chart(fig2)

    fig3 = px.box(
        data,
        x="output",
        y="trtbps",
        title="Blood Pressure Distribution"
    )
    st.plotly_chart(fig3)

    fig4 = px.histogram(
        data,
        x="cp",
        color="output",
        title="Chest Pain Analysis"
    )
    st.plotly_chart(fig4)

# -------------------- Admin Page --------------------

elif selected == "Admin":

    st.title("👨‍⚕️ Admin Dashboard")

    conn = sqlite3.connect("hospital.db")
    cursor = conn.cursor()

    # ---------------- High Risk Patients ----------------

    st.subheader("⚠ High Risk Patients")

    high = pd.read_sql("SELECT * FROM high_risk", conn)

    # Column headings
    h1,h2,h3,h4,h5,h6 = st.columns(6)
    h1.markdown("**ID**")
    h2.markdown("**Name**")
    h3.markdown("**Age**")
    h4.markdown("**Blood Pressure**")
    h5.markdown("**Cholesterol**")
    h6.markdown("**Action**")

    for index,row in high.iterrows():

        col1,col2,col3,col4,col5,col6 = st.columns(6)

        col1.write(row["id"])
        col2.write(row["name"])
        col3.write(row["age"])
        col4.write(row["bp"])
        col5.write(row["chol"])

        if col6.button("Delete", key=f"high{row['id']}"):

            cursor.execute(
                "DELETE FROM high_risk WHERE id=?",
                (row["id"],)
            )

            conn.commit()
            st.success("Patient Deleted")
            st.rerun()

    # ---------------- Low Risk Patients ----------------

    st.subheader("✅ Low Risk Patients")

    low = pd.read_sql("SELECT * FROM low_risk", conn)

    # Column headings
    l1,l2,l3,l4,l5,l6 = st.columns(6)
    l1.markdown("**ID**")
    l2.markdown("**Name**")
    l3.markdown("**Age**")
    l4.markdown("**Blood Pressure**")
    l5.markdown("**Cholesterol**")
    l6.markdown("**Action**")

    for index,row in low.iterrows():

        col1,col2,col3,col4,col5,col6 = st.columns(6)

        col1.write(row["id"])
        col2.write(row["name"])
        col3.write(row["age"])
        col4.write(row["bp"])
        col5.write(row["chol"])

        if col6.button("Delete", key=f"low{row['id']}"):

            cursor.execute(
                "DELETE FROM low_risk WHERE id=?",
                (row["id"],)
            )

            conn.commit()
            st.success("Patient Deleted")
            st.rerun()

    conn.close()