from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Cargar modelo Keras entrenado en Colab
model = tf.keras.models.load_model("student_performance_model.h5")

# Mapeo de clases
CLASS_LABELS = {
    0: "bajo",
    1: "medio",
    2: "alto"
}

# Categorías usadas en el entrenamiento (DEBEN COINCIDIR CON EL NOTEBOOK)
GENDER_CATS = ["female", "male"]
RACE_CATS = ["group A", "group B", "group C", "group D", "group E"]
PARENT_EDU_CATS = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]
LUNCH_CATS = ["free/reduced", "standard"]
PREP_CATS = ["none", "completed"]

def one_hot(value, categories):
    """Devuelve un vector one-hot para 'value' dado un listado de categories."""
    vec = [0] * len(categories)
    if value in categories:
        idx = categories.index(value)
        vec[idx] = 1
    return vec

def encode_input(data):
    """
    data: dict con las claves:
    - gender
    - race_ethnicity
    - parental_level_of_education
    - lunch
    - test_preparation_course
    """
    gender_vec = one_hot(data["gender"], GENDER_CATS)
    race_vec = one_hot(data["race_ethnicity"], RACE_CATS)
    parent_vec = one_hot(data["parental_level_of_education"], PARENT_EDU_CATS)
    lunch_vec = one_hot(data["lunch"], LUNCH_CATS)
    prep_vec = one_hot(data["test_preparation_course"], PREP_CATS)

    full_vec = gender_vec + race_vec + parent_vec + lunch_vec + prep_vec
    return np.array(full_vec, dtype=float).reshape(1, -1)

@app.route("/")
def index():
    return jsonify({"message": "API de predicción de rendimiento académico activa"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Falta el campo requerido: {field}"}), 400

        x_input = encode_input(data)  # vector 1 x N
        preds = model.predict(x_input)
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
