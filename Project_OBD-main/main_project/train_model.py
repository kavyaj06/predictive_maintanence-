import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

def load_and_prepare_data():
    """Load engine data and prepare features"""
    print("Loading engine data...")
    
    # Load the main engine dataset
    df = pd.read_csv('Datasets/engine_data.csv')
    print(f"Loaded {len(df)} samples")
    print(f"Features: {list(df.columns)}")
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Remove any rows with missing values
    df = df.dropna()
    print(f"After removing missing values: {len(df)} samples")
    
    # Separate features and target
    feature_columns = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                      'Coolant pressure', 'lub oil temp', 'Coolant temp']
    
    X = df[feature_columns]
    y = df['Engine Condition']
    
    print(f"Feature shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_columns

def add_quick_features(X):
    """Add a few key engineered features for better performance"""
    print("\nAdding quick engineered features...")
    
    X_enhanced = X.copy()
    
    # Add temperature ratio (important for engine health)
    X_enhanced['temp_ratio'] = X_enhanced['lub oil temp'] / (X_enhanced['Coolant temp'] + 1e-8)
    
    # Add pressure efficiency indicator
    X_enhanced['pressure_efficiency'] = X_enhanced['Fuel pressure'] / (X_enhanced['Lub oil pressure'] + 1e-8)
    
    # Add engine load indicator
    X_enhanced['engine_load'] = X_enhanced['Engine rpm'] * X_enhanced['Fuel pressure'] / 1000
    
    print(f"Enhanced features shape: {X_enhanced.shape}")
    return X_enhanced

def train_model(X, y):
    """Train Random Forest model with optimized parameters"""
    print("\nTraining optimized Random Forest model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try both Random Forest and Gradient Boosting, pick the best
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    # Choose the best model
    if gb_accuracy > rf_accuracy:
        print(f"Gradient Boosting performs better: {gb_accuracy:.4f} vs {rf_accuracy:.4f}")
        final_model = gb_model
        y_pred = gb_pred
        accuracy = gb_accuracy
        model_name = "GradientBoosting"
    else:
        print(f"Random Forest performs better: {rf_accuracy:.4f} vs {gb_accuracy:.4f}")
        final_model = rf_model
        y_pred = rf_pred
        accuracy = rf_accuracy
        model_name = "RandomForest"
    
    print(f"\nModel Performance ({model_name}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return final_model, scaler, X_train.columns.tolist()

def save_model_and_scaler(model, scaler, feature_names):
    """Save the trained model and scaler"""
    print("\nSaving model and scaler...")
    
    # Save the model
    with open('random_forest_model-2.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names for reference
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print("Model, scaler, and feature names saved successfully!")

def generate_sample_predictions(model, scaler, feature_names):
    """Generate 2 sample predictions with confidence scores"""
    print("\nGenerating sample predictions...")
    
    # Sample 1: Good engine condition (6 original features)
    sample1_original = np.array([[2200, 3.2, 45.0, 1.1, 85.0, 90.0]]).reshape(1, -1)
    
    # Add engineered features for sample1
    temp_ratio1 = sample1_original[0][4] / (sample1_original[0][5] + 1e-8)
    pressure_efficiency1 = sample1_original[0][2] / (sample1_original[0][1] + 1e-8)
    engine_load1 = sample1_original[0][0] * sample1_original[0][2] / 1000
    
    sample1 = np.array([[2200, 3.2, 45.0, 1.1, 85.0, 90.0, temp_ratio1, pressure_efficiency1, engine_load1]]).reshape(1, -1)
    sample1_scaled = scaler.transform(sample1)
    pred1 = model.predict(sample1_scaled)[0]
    prob1 = model.predict_proba(sample1_scaled)[0]
    
    print(f"\nSample 1 - Good Engine:")
    print(f"Features: Engine rpm=2200, Lub oil pressure=3.2, Fuel pressure=45.0, Coolant pressure=1.1, lub oil temp=85.0, Coolant temp=90.0")
    print(f"Prediction: {'Good' if pred1 == 1 else 'Bad'}")
    print(f"Confidence: {max(prob1):.4f}")
    print(f"Probabilities: [Bad: {prob1[0]:.4f}, Good: {prob1[1]:.4f}]")
    
    # Sample 2: Bad engine condition (6 original features)
    sample2_original = np.array([[500, 1.5, 20.0, 0.5, 95.0, 110.0]]).reshape(1, -1)
    
    # Add engineered features for sample2
    temp_ratio2 = sample2_original[0][4] / (sample2_original[0][5] + 1e-8)
    pressure_efficiency2 = sample2_original[0][2] / (sample2_original[0][1] + 1e-8)
    engine_load2 = sample2_original[0][0] * sample2_original[0][2] / 1000
    
    sample2 = np.array([[500, 1.5, 20.0, 0.5, 95.0, 110.0, temp_ratio2, pressure_efficiency2, engine_load2]]).reshape(1, -1)
    sample2_scaled = scaler.transform(sample2)
    pred2 = model.predict(sample2_scaled)[0]
    prob2 = model.predict_proba(sample2_scaled)[0]
    
    print(f"\nSample 2 - Bad Engine:")
    print(f"Features: Engine rpm=500, Lub oil pressure=1.5, Fuel pressure=20.0, Coolant pressure=0.5, lub oil temp=95.0, Coolant temp=110.0")
    print(f"Prediction: {'Good' if pred2 == 1 else 'Bad'}")
    print(f"Confidence: {max(prob2):.4f}")
    print(f"Probabilities: [Bad: {prob2[0]:.4f}, Good: {prob2[1]:.4f}]")
    
    return sample1_original[0], sample2_original[0]

def main():
    """Main training pipeline"""
    print("=== Vehicle Engine Condition Prediction Model Training (Optimized) ===\n")
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data()
    
    # Add quick engineered features
    X_enhanced = add_quick_features(X)
    
    # Train the model
    model, scaler, feature_names = train_model(X_enhanced, y)
    
    # Save model and scaler
    save_model_and_scaler(model, scaler, feature_names)
    
    # Generate sample predictions
    sample1, sample2 = generate_sample_predictions(model, scaler, feature_names)
    
    print(f"\n=== Training Complete ===")
    print(f"Model saved as: random_forest_model-2.pkl")
    print(f"Scaler saved as: scaler.pkl")
    print(f"Feature names saved as: feature_names.pkl")
    print(f"\nUse these sample data points in your UI:")
    print(f"Sample 1 (Good): {dict(zip(feature_names[:6], sample1))}")
    print(f"Sample 2 (Bad): {dict(zip(feature_names[:6], sample2))}")

if __name__ == "__main__":
    main()