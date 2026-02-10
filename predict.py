import joblib
import pandas as pd
import numpy as np

def predecir_supervivencia():
    # 1. CARGAR EL CEREBRO
    print("⏳ Cargando modelo...")
    try:
        # Carga el objeto completo (incluye los nombres de las columnas aprendidas)
        model = joblib.load('models/titanic_logistic_v1.pkl')
        print("✅ Modelo cargado.")
    except FileNotFoundError:
        print("❌ ERROR: No encuentro el archivo .pkl")
        return

    # 2. DEFINIR DATOS DE PRUEBA (CASOS HIPOTÉTICOS)
    # Imaginemos 2 pasajeros nuevos que llegan a la taquilla:
    # Pasajero 0: Pobre, Hombre, Joven (Debería morir 💀)
    # Pasajero 1: Rica, Mujer, Adulta (Debería vivir 🟢)
    
    input_data = {
        'Pclass': [3, 1],
        'Age': [22, 35],
        'SibSp': [0, 1],
        'Parch': [0, 0],
        'Fare': [7.25, 71.28],
        # OJO: Aquí simulamos el One-Hot Encoding MANUALMENTE
        # En el futuro usaremos Pipelines para automatizar esto,
        # pero hoy lo hacemos a mano para que entiendas el dolor.
        'Sex_female': [0, 1],  # 0=Hombre, 1=Mujer
        'Sex_male': [1, 0],    # Redundante, pero si el modelo lo usó, lo necesita
        'Embarked_Q': [0, 0],
        'Embarked_S': [1, 0],  # El primero embarcó en S
        'Title_Miss': [0, 0],
        'Title_Mr': [1, 0],    # El primero es Mr
        'Title_Mrs': [0, 1],   # La segunda es Mrs
        # Nota: Faltan muchas columnas (Title_Other, etc...)
    }
    
    df_input = pd.DataFrame(input_data)

    # 3. EL TRUCO DE MAGIA (ALINEACIÓN DE COLUMNAS)
    # Obtenemos la lista exacta de columnas que el modelo "memorizó"
    expected_cols = model.feature_names_in_
    
    # Reindexamos: Esto crea las columnas que faltan y las llena con 0
    # Y ordena las columnas exactamente como el modelo las quiere.
    df_final = df_input.reindex(columns=expected_cols, fill_value=0)

    print(f"\n📋 Datos procesados ({df_final.shape[0]} pasajeros, {df_final.shape[1]} columnas)")
    
    # 4. PREDICCIÓN
    predicciones = model.predict(df_final)
    probs = model.predict_proba(df_final)[:, 1] # Probabilidad de clase 1 (Sobrevivir)

    print("\n🔮 RESULTADOS DEL ORÁCULO:")
    print("-" * 30)
    for i, (pred, prob) in enumerate(zip(predicciones, probs)):
        status = "🟢 SOBREVIVE" if pred == 1 else "💀 MUERE"
        print(f"Pasajero {i+1}: {status} (Probabilidad: {prob:.1%})")
    print("-" * 30)

if __name__ == "__main__":
    predecir_supervivencia()

# Me tengo que asegurar que estoy en ds_bootcamp. Sino, conda activate ds_bootcamp