import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. CARGAR EL NUEVO MODELO (RANDOM FOREST)
@st.cache_resource
def load_model():
    # Asegúrate de que este archivo existe en la carpeta models/
    return joblib.load('models/titanic_random_forest_v1.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.info("Asegúrate de haber ejecutado el notebook y guardado el modelo como 'titanic_random_forest_v1.pkl'")
    st.stop()

# 2. TÍTULO Y DESCRIPCIÓN
st.title("🚢 Titanic Survival Predictor (Random Forest)")
st.markdown("""
**¿Habrías sobrevivido al Titanic?** Este modelo usa **Random Forest** (un bosque de decisiones) con una precisión estimada del **85%**.
Ahora analiza factores complejos como el tamaño de tu familia y el precio real que pagaste por persona.
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
# --- AQUÍ ESTÁ LA MAGIA MATEMÁTICA ---
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
# Evitamos dividir por cero (aunque family_size mínimo es 1, es buena práctica)
fare_per_person = fare / family_size if family_size > 0 else fare

# Construimos el diccionario de entrada
input_data = {
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    
    # NUEVAS VARIABLES CALCULADAS
    'FamilySize': [family_size],
    'IsAlone': [is_alone],
    'FarePerPerson': [fare_per_person],
    
    # VARIABLES DUMMY (Inicializamos a 0)
    'Sex_male': [0],
    'Sex_female': [0],
    'Embarked_S': [0],
    'Embarked_C': [0],
    'Embarked_Q': [0],
    
    # TÍTULOS (Simplificado para la App)
    'Title_Mr': [1 if sex == "Hombre" else 0],
    'Title_Mrs': [1 if sex == "Mujer" and age > 25 else 0],
    'Title_Miss': [1 if sex == "Mujer" and age <= 25 else 0],
    'Title_Master': [1 if sex == "Hombre" and age < 10 else 0],
}

# Lógica condicional para Dummies
if sex == "Hombre":
    input_data['Sex_male'] = [1]
else:
    input_data['Sex_female'] = [1]

if "S" in embarked: input_data['Embarked_S'] = [1]
elif "C" in embarked: input_data['Embarked_C'] = [1]
else: input_data['Embarked_Q'] = [1]

# Convertir a DataFrame
df_input = pd.DataFrame(input_data)

# 5. ALINEAR COLUMNAS (El truco maestro)
# Esto asegura que el orden de las columnas sea EXACTAMENTE el que aprendió el modelo
try:
    expected_cols = model.feature_names_in_
    df_final = df_input.reindex(columns=expected_cols, fill_value=0)
except AttributeError:
    # Si el modelo es viejo y no tiene feature_names_in_, usamos el df tal cual (riesgoso)
    st.warning("⚠️ El modelo no tiene lista de características guardada. Usando orden por defecto.")
    df_final = df_input

# 6. BOTÓN DE PREDICCIÓN
if st.button("🔮 Calcular Probabilidad"):
    prediction = model.predict(df_final)[0]
    probability = model.predict_proba(df_final)[0][1] * 100
    
    st.divider()
    
    if prediction == 1:
        st.success(f"🟢 **¡SOBREVIVES!**")
        st.balloons()
        st.write(f"El Random Forest calcula que tienes un **{probability:.1f}%** de probabilidad de sobrevivir.")
        st.info(f"💡 Dato Curioso: Viajas con una familia de {family_size} personas y pagaste £{fare_per_person:.1f} por cabeza.")
    else:
        st.error(f"💀 **NO SOBREVIVES**")
        st.write(f"La probabilidad de supervivencia es baja: **{probability:.1f}%**.")
        st.write("En 1912, la prioridad era 'Mujeres y niños primero' y las clases altas.")