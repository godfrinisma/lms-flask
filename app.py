from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Load model and scaler
rf_model = joblib.load('model4.pkl')
scaler = joblib.load('scaler3.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    print(features)
    #scaled_features = scaler.transform(features)
    s = StandardScaler()
    scaled_features = s.fit_transform(features)
    prediction = rf_model.predict(scaled_features)
    print(prediction)
    return jsonify({'attention': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
