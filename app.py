eimport pandas as pd
import joblib
from flask import Flask, request, jsonify
from xgboost import XGBRegressor

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

FEATURES = ['city', 'province', 'latitude', 'longitude', 'lease_term', 'type', 'beds', 'baths', 'sq_feet',
            'furnishing', 'availability_date', 'smoking', 'cats', 'dogs', 'Walk Score', 'Bike Score', 'studio']
@app.route('/')
def index():
    return "Welcome to the Housing Price Prediction API. Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df_input = pd.DataFrame([data])
        for col, le in encoders.items():
            if col in df_input:
                df_input[col] = le.transform([df_input[col].strip().lower()])[0]
        df_input = df_input[FEATURES]
        prediction = model.predict(df_input)[0]
        return jsonify({'predicted_price': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
