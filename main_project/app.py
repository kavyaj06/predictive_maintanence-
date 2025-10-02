from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load model and scaler
with open("random_forest_model-2.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the 6 basic features that the UI will send
BASIC_FEATURES = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure", 
    "Coolant pressure", "lub oil temp", "Coolant temp"
]

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        
        # Validate required basic features
        missing = [name for name in BASIC_FEATURES if name not in payload]
        if missing:
            return jsonify({"error": "Missing feature values", "missing": missing}), 400
        
        # Extract basic features
        engine_rpm = float(payload["Engine rpm"])
        lub_oil_pressure = float(payload["Lub oil pressure"])
        fuel_pressure = float(payload["Fuel pressure"])
        coolant_pressure = float(payload["Coolant pressure"])
        lub_oil_temp = float(payload["lub oil temp"])
        coolant_temp = float(payload["Coolant temp"])
        
        # Calculate engineered features (same as in training)
        temp_ratio = lub_oil_temp / (coolant_temp + 1e-8)
        pressure_efficiency = fuel_pressure / (lub_oil_pressure + 1e-8)
        engine_load = engine_rpm * fuel_pressure / 1000
        
        # Create input array with all 9 features in correct order
        input_row = np.array([[
            engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure,
            lub_oil_temp, coolant_temp, temp_ratio, pressure_efficiency, engine_load
        ]])
        
        # Apply the trained scaler
        input_row_scaled = scaler.transform(input_row)
        
        # Make prediction
        pred = model.predict(input_row_scaled)
        pred_value = int(pred[0])
        
        # Rule-based override for obviously bad conditions
        # These thresholds are based on the sample_data_bad_condition.txt
        rule_based_bad = (
            lub_oil_pressure < 1.5 or  # Very low oil pressure
            coolant_temp > 105 or      # Overheating
            lub_oil_temp > 95 or       # High oil temperature
            engine_rpm < 600 or        # Very low RPM
            fuel_pressure < 20 or      # Low fuel pressure
            coolant_pressure < 0.8     # Low coolant pressure
        )
        
        # Override ML prediction if rule-based system detects bad condition
        if rule_based_bad and pred_value == 1:
            pred_value = 0  # Force to bad condition
            print(f"Rule-based override: ML predicted good but conditions are clearly bad")
        
        engine_condition = "Engine Condition is Good" if pred_value == 1 else "Engine Condition is Bad"
        
        # Get prediction probabilities for confidence
        pred_proba = model.predict_proba(input_row_scaled)[0]
        confidence = max(pred_proba)
        
        return jsonify({
            "Engine_Condition": engine_condition,
            "prediction": pred_value,
            "confidence": float(confidence),
            "probabilities": {
                "bad": float(pred_proba[0]),
                "good": float(pred_proba[1])
            }
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


