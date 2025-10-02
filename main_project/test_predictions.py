import pickle
import numpy as np

# Load the trained model and scaler
with open('random_forest_model-2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("=== Vehicle Engine Condition Prediction Samples ===\n")

# Sample 1: Good engine condition (from training output)
sample1_original = np.array([[2200, 3.2, 45.0, 1.1, 85.0, 90.0]])

# Add engineered features for sample1
temp_ratio1 = sample1_original[0][4] / (sample1_original[0][5] + 1e-8)
pressure_efficiency1 = sample1_original[0][2] / (sample1_original[0][1] + 1e-8)
engine_load1 = sample1_original[0][0] * sample1_original[0][2] / 1000

sample1 = np.array([[2200, 3.2, 45.0, 1.1, 85.0, 90.0, temp_ratio1, pressure_efficiency1, engine_load1]])
sample1_scaled = scaler.transform(sample1)
pred1 = model.predict(sample1_scaled)[0]
prob1 = model.predict_proba(sample1_scaled)[0]

print("Sample 1 - Good Engine:")
print(f"Features: Engine rpm=2200, Lub oil pressure=3.2, Fuel pressure=45.0, Coolant pressure=1.1, lub oil temp=85.0, Coolant temp=90.0")
print(f"Prediction: {'Good' if pred1 == 1 else 'Bad'}")
print(f"Confidence: {max(prob1):.4f} ({max(prob1)*100:.1f}%)")
print(f"Probabilities: [Bad: {prob1[0]:.4f} ({prob1[0]*100:.1f}%), Good: {prob1[1]:.4f} ({prob1[1]*100:.1f}%)]")
print()

# Sample 2: Bad engine condition (from training output)
sample2_original = np.array([[500, 1.5, 20.0, 0.5, 95.0, 110.0]])

# Add engineered features for sample2
temp_ratio2 = sample2_original[0][4] / (sample2_original[0][5] + 1e-8)
pressure_efficiency2 = sample2_original[0][2] / (sample2_original[0][1] + 1e-8)
engine_load2 = sample2_original[0][0] * sample2_original[0][2] / 1000

sample2 = np.array([[500, 1.5, 20.0, 0.5, 95.0, 110.0, temp_ratio2, pressure_efficiency2, engine_load2]])
sample2_scaled = scaler.transform(sample2)
pred2 = model.predict(sample2_scaled)[0]
prob2 = model.predict_proba(sample2_scaled)[0]

print("Sample 2 - Bad Engine:")
print(f"Features: Engine rpm=500, Lub oil pressure=1.5, Fuel pressure=20.0, Coolant pressure=0.5, lub oil temp=95.0, Coolant temp=110.0")
print(f"Prediction: {'Good' if pred2 == 1 else 'Bad'}")
print(f"Confidence: {max(prob2):.4f} ({max(prob2)*100:.1f}%)")
print(f"Probabilities: [Bad: {prob2[0]:.4f} ({prob2[0]*100:.1f}%), Good: {prob2[1]:.4f} ({prob2[1]*100:.1f}%)]")
print()

# Additional sample: Very good engine
sample3_original = np.array([[2000, 4.0, 50.0, 1.5, 80.0, 85.0]])

# Add engineered features for sample3
temp_ratio3 = sample3_original[0][4] / (sample3_original[0][5] + 1e-8)
pressure_efficiency3 = sample3_original[0][2] / (sample3_original[0][1] + 1e-8)
engine_load3 = sample3_original[0][0] * sample3_original[0][2] / 1000

sample3 = np.array([[2000, 4.0, 50.0, 1.5, 80.0, 85.0, temp_ratio3, pressure_efficiency3, engine_load3]])
sample3_scaled = scaler.transform(sample3)
pred3 = model.predict(sample3_scaled)[0]
prob3 = model.predict_proba(sample3_scaled)[0]

print("Sample 3 - Very Good Engine:")
print(f"Features: Engine rpm=2000, Lub oil pressure=4.0, Fuel pressure=50.0, Coolant pressure=1.5, lub oil temp=80.0, Coolant temp=85.0")
print(f"Prediction: {'Good' if pred3 == 1 else 'Bad'}")
print(f"Confidence: {max(prob3):.4f} ({max(prob3)*100:.1f}%)")
print(f"Probabilities: [Bad: {prob3[0]:.4f} ({prob3[0]*100:.1f}%), Good: {prob3[1]:.4f} ({prob3[1]*100:.1f}%)]")
print()

print("=== Model Performance Summary ===")
print(f"Model accuracy: 66.37%")
print(f"Features used: {feature_names}")
print(f"Training samples: 19,535")
print(f"Test samples: 3,907")
