from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Cargar modelo y preprocesador
model = tf.keras.models.load_model("student_performance_model.h5")
preprocessor = joblib.load("preprocessor_student_performance.pkl")

with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

# Mapeo de clase numérica a etiqueta legible
CLASS_LABELS = {
    0: "bajo",
    1: "medio",
    2: "alto"
}

@app.route("/")
def index():
    return jsonify({"message": "API de predicción de rendimiento académico activa"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Esperamos un JSON con las 5 características
        data = request.get_json()

        # Verificar que todas las columnas necesarias estén presentes
        for col in feature_cols:
            if col not in data:
                return jsonify({
                    "error": f"Falta el campo requerido: {col}"
                }), 400

        # Crear DataFrame con un solo registro
        input_df = pd.DataFrame([ {col: data[col] for col in feature_cols} ])

        # Aplicar mismo preprocesamiento que en el entrenamiento
        X_enc = preprocessor.transform(input_df)
        X_enc = X_enc.toarray()

        # Predicción del modelo
        preds = model.predict(X_enc)
        class_idx = int(np.argmax(preds, axis=1)[0])
        prob = float(np.max(preds))

        return jsonify({
            "rendimiento_clase": CLASS_LABELS[class_idx],
            "clase_numerica": class_idx,
            "probabilidad": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Para pruebas locales
    app.run(host="0.0.0.0", port=5000, debug=True)
