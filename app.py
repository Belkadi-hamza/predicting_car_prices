# api.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load('model/lgb_model.pkl')

def clean_kilometrage(value):
    """Preprocess kilometrage field"""
    try:
        if pd.isna(value):
            return np.nan
            
        str_value = str(value).replace(' ', '').upper()
        
        if 'PLUSDE' in str_value:
            return 500001
            
        numbers = [int(num) for num in str_value.split('-') if num.strip().isdigit()]
        
        if len(numbers) == 2:
            return (numbers[0] + numbers[1]) // 2
        elif len(numbers) == 1:
            return numbers[0]
        return np.nan
    except:
        return np.nan

def clean_puissance(value):
    """Preprocess puissance_fiscale field"""
    try:
        if pd.isna(value):
            return np.nan
            
        if 'Plus de' in str(value):
            return int(value.split()[-2]) + 1
        return int(''.join(filter(str.isdigit, str(value))))
    except:
        return np.nan

def encodage(df):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    categorical_cols = []
    features = df.columns.values.tolist()
    for col in features:
        if df[col].dtype in numerics: continue
        categorical_cols.append(col)
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(list(df[col].astype(str).values))
            df[col] = le.transform(list(df[col].astype(str).values))
    return df

def scaler(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    return df

@app.route('/',methods=['GET'])
def home():
    return "Welcome to API Car prediction"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # 2. Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # 3. Clean numerical fields
        if 'kilométrage' in input_df.columns:
            input_df['kilométrage'] = input_df['kilométrage'].apply(clean_kilometrage)
        
        if 'puissance_fiscale' in input_df.columns:
            input_df['puissance_fiscale'] = input_df['puissance_fiscale'].apply(clean_puissance)
        
        # 4. Encode categorical features
        input_df = encodage(input_df)
        
        # 5. Scale features
        input_df = scaler(input_df)
        
        # 6. Make prediction
        prediction = model.predict(input_df, num_iteration=model.best_iteration)
        return jsonify({
            'predicted_price': float(prediction[0]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)