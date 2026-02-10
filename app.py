import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. CARGAR EL MODELO (Con caché para que sea rápido)
@st.cache_resource
def load_model():
    return joblib.load('models/titanic_logistic_v1.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

# 2. TÍTULO Y DESCRIPCIÓN
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
**¿Habrías sobrevivido al Titanic?** Introduce tus datos y deja que el algoritmo de Machine Learning (Regresión Logística - 82% Acc) decida tu destino.
""")

# 3. BARRA LATERAL (INPUTS DEL USUARIO)
st.sidebar.header("Tus Datos de Pasajero")

# Variables numéricas
pclass = st.sidebar.radio("Clase del Billete", [1, 2, 3], index=2, help="1ª = Alta, 3ª = Baja")
sex = st.sidebar.radio("Sexo", ["Hombre", "Mujer"], index=0)
age = st.sidebar.slider("Edad", 0, 100, 30)
sibsp = st.sidebar.slider("Hermanos/Esposos a bordo", 0, 8, 0)
parch = st.sidebar.slider("Padres/Hijos a bordo", 0, 6, 0)
fare = st.sidebar.number_input("Precio del Billete (£)", 0.0, 512.0, 32.0)
embarked = st.sidebar.selectbox("Puerto de Embarque", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# 4. PREPARAR DATOS (Feature Engineering en vivo)
# Convertimos el texto del usuario a lo que el modelo entiende
input_data = {
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [0],    # Inicializamos a 0
    'Sex_female': [0],  # Inicializamos a 0
    'Embarked_S': [0],
    'Embarked_C': [0],
    'Embarked_Q': [0],
    # Asumimos títulos básicos por simplicidad en la UI
    'Title_Mr': [1 if sex == "Hombre" else 0],
    'Title_Mrs': [1 if sex == "Mujer" and age > 25 else 0],
    'Title_Miss': [1 if sex == "Mujer" and age <= 25 else 0],
    'Title_Master': [1 if sex == "Hombre" and age < 10 else 0],
}

# Lógica condicional simple para rellenar los One-Hot
if sex == "Hombre":
    input_data['Sex_male'] = [1]
else:
    input_data['Sex_female'] = [1]

if "S" in embarked: input_data['Embarked_S'] = [1]
elif "C" in embarked: input_data['Embarked_C'] = [1]
else: input_data['Embarked_Q'] = [1]

# Convertir a DataFrame
df_input = pd.DataFrame(input_data)

# 5. ALINEAR COLUMNAS (El truco maestro de ingeniería)
# Aseguramos que el input tenga EXACTAMENTE las mismas columnas que el modelo vio al entrenar
expected_cols = model.feature_names_in_
df_final = df_input.reindex(columns=expected_cols, fill_value=0)

# 6. BOTÓN DE PREDICCIÓN
if st.button("🔮 Calcular Probabilidad"):
    prediction = model.predict(df_final)[0]
    probability = model.predict_proba(df_final)[0][1] * 100
    
    st.divider()
    
    if prediction == 1:
        st.success(f"🟢 **¡SOBREVIVES!**")
        st.balloons()
        st.write(f"El algoritmo calcula que tienes un **{probability:.1f}%** de probabilidad de sobrevivir.")
        st.write("Tu estatus socioeconómico y edad jugaron a tu favor.")
    else:
        st.error(f"💀 **NO SOBREVIVES**")
        st.write(f"La probabilidad de supervivencia es baja: **{probability:.1f}%**.")
        st.write("En 1912, la prioridad era 'Mujeres y niños primero' y las clases altas.")