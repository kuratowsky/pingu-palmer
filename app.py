import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- 1. CÀRREGA DE COMPONENTS EN ARRENCAR EL SERVIDOR ---
try:
    # Hem de carregar els dos components de preprocessament
    with open('models/dict_vectorizer.pkl', 'rb') as f:
        dict_vectorizer = pickle.load(f)
    with open('models/standard_scaler.pkl', 'rb') as f:
        standard_scaler = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    MODELS = {
        'log_reg': pickle.load(open('models/log_reg_model.pkl', 'rb')),
        'svm': pickle.load(open('models/svm_model.pkl', 'rb')),
        'dt': pickle.load(open('models/dt_model.pkl', 'rb')),
        'knn': pickle.load(open('models/knn_model.pkl', 'rb')),
    }
    
    MODEL_NAMES = {
        'log_reg': 'Regressió Logística',
        'svm': 'Màquina de Suport Vectorial (SVM)',
        'dt': 'Arbre de Decisió',
        'knn': 'K Veïns Més Propers (KNN)',
    }
    SPECIES_CLASSES = le.classes_ 

    print("✅ Components de ML carregats amb èxit (DictVectorizer i StandardScaler).")
except Exception as e:
    print(f"❌ ERROR carregant models: {e}")
    dict_vectorizer, standard_scaler, MODELS, SPECIES_CLASSES = None, None, None, None

# --- 2. ENDPOINT DE PREDICCIÓ ---
@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if not MODELS or model_name not in MODELS:         
        return jsonify({"error": "Model no disponible"}), 404

    try:
        data = request.get_json(force=True)
        new_penguin_df = pd.DataFrame([data])
        
        # Columnes utilitzades durant l'entrenament
        categorical_cols = ['island', 'sex']
        numerical_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

        # 1. Preprocessament Categòric (DictVectorizer)
        X_cat_dict = new_penguin_df[categorical_cols].to_dict('records')
        X_cat_encoded = dict_vectorizer.transform(X_cat_dict) # Només transform!
        
        # 2. Preprocessament Numèric (StandardScaler)
        X_num_scaled = standard_scaler.transform(new_penguin_df[numerical_cols]) # Només transform!
        
        # 3. Combinació de les dades (Igual que np.hstack en l'entrenament)
        X_new_processed = np.hstack([X_cat_encoded, X_num_scaled])
        
        # 4. Fer la predicció
        model = MODELS[model_name]
        prediction_index = model.predict(X_new_processed)[0]
        prediction_species = SPECIES_CLASSES[prediction_index]
        
        response = {
            "model_solicitat": MODEL_NAMES[model_name],
            "dades_entrada": data,
            "prediccio_index": int(prediction_index),
            "prediccio_especie": prediction_species
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Error processant la petició: {str(e)}", 
                        "model_solicitat": model_name}), 500

if __name__ == '__main__':
    print("📢 Iniciant el servidor Flask al port 5000...")
    app.run(host='0.0.0.0', port=5000)