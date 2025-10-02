# 🚛 Predictive Maintenance System for Commercial Vehicles

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent **predictive maintenance system** designed for commercial vehicle fleets that uses machine learning to analyze engine telemetry data and predict potential maintenance issues before they become critical failures.


## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🛠️ Technologies Used](#️-technologies-used)
- [📊 Model](#-model-performance)
- [🏗️ Project Structure](#️-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 Usage](#-usage)
- [📈 Features](#-features)

## DISCLAIMER: TO USE THE ENTIRE APPLICATION, GET TO [HUGGINGFACE](https://huggingface.co/chari-00/Project_OBD)
- ML Model is present only in huggingface and not here (main_project\random_forest_model-2.pkl)

## 🎯 Project Overview

This system addresses the critical need for **proactive vehicle maintenance** in commercial fleets by:

- **Real-time Engine Analysis**: Monitors 6 key engine parameters (RPM, pressures, temperatures)
- **Predictive Analytics**: Uses machine learning to identify potential issues before breakdown
- **Cost Reduction**: Prevents expensive emergency repairs and reduces vehicle downtime
- **Fleet Optimization**: Helps fleet managers schedule maintenance efficiently
- **Safety Enhancement**: Reduces risk of vehicle failures during operation

### 🎯 Target Use Cases
- **Commercial Fleet Management**: Trucking companies, delivery services
- **Heavy Equipment Monitoring**: Construction, mining, agriculture
- **Public Transportation**: Bus fleets, municipal vehicles
- **Logistics Companies**: Last-mile delivery, freight transport

## 🛠️ Technologies Used

### **Backend & Machine Learning**
- **Python 3.8+** - Core programming language
- **Flask 3.1.2** - Web framework for REST API
- **scikit-learn 1.7.2** - Machine learning algorithms
- **NumPy 2.3.3** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Pickle** - Model serialization

### **Frontend & UI**
- **HTML5** - Structure and semantic markup
- **CSS3** - Modern styling with CSS Grid/Flexbox
- **JavaScript (ES6+)** - Interactive functionality
- **Google Fonts (Inter)** - Typography
- **Responsive Design** - Mobile-first approach

### **Machine Learning Pipeline**
- **Random Forest Classifier** - Primary prediction algorithm
- **Gradient Boosting Classifier** - Alternative model for comparison
- **StandardScaler** - Feature normalization
- **Feature Engineering** - Custom engineered features for better accuracy
- **Cross-validation** - Model validation and testing


## 📊 Model

### **Features Used**
**Original Features (6):**
1. Engine RPM
2. Lub Oil Pressure
3. Fuel Pressure
4. Coolant Pressure
5. Lub Oil Temperature
6. Coolant Temperature

**Engineered Features (3):**
7. Temperature Ratio (oil temp / coolant temp)
8. Pressure Efficiency (fuel pressure / oil pressure)
9. Engine Load (RPM × fuel pressure / 1000)


## 🏗️ Project Structure

```
Project_OBD/
├── 📄 index.html                          # Main web interface
├── 📄 README.md                           # Project documentation
├── 📄 model_accuracy_report.txt           # Detailed model performance
├── 📄 sample_data_good_condition.txt      # Test data for healthy engines
├── 📄 sample_data_bad_condition.txt       # Test data for problematic engines
│
├── 📁 main_project/                       # Core application
│   ├── 🐍 app.py                         # Flask backend API
│   ├── 🐍 train_model.py                 # Model training script
│   ├── 🐍 test_predictions.py            # Model testing utilities
│   ├── 📄 requirements.txt               # Python dependencies
│   ├── 🤖 random_forest_model-2.pkl      # Trained ML model
│   ├── ⚙️ scaler.pkl                     # Feature scaler
│   ├── 📋 feature_names.pkl              # Feature reference
│   │
│   ├── 📁 Datasets/                      # Training data
│   │   └── 📊 engine_data.csv            # Engine condition dataset
│   │
│   └── 📁 UI/                            # Additional UI components
│       ├── 📄 FormRemodeled.html
│       ├── 📄 FormUI.html
│       └── 📁 project/                   # React/TypeScript components
│
└── 📁 venv/                              # Virtual environment (excluded from git)
```

## ⚙️ Installation & Setup

### **Prerequisites**
- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **Web browser** (Chrome, Firefox, Safari, Edge)

### **Step 1: Clone the Repository**
```bash
# Clone the repository
git clone https://github.com/yourusername/Project_OBD.git

# Navigate to project directory
cd Project_OBD
```

### **Step 2: Set Up Python Virtual Environment**

**On Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
# Navigate to main project directory
cd main_project

# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

### **Step 4: Verify Model Files**
Ensure these files exist in `main_project/`:
- ✅ `random_forest_model-2.pkl`
- ✅ `scaler.pkl`
- ✅ `feature_names.pkl`

If missing, run the training script:
```bash
python train_model.py
```

## 🚀 Usage

### **Starting the Application**

1. **Activate Virtual Environment** (if not already active):
   ```bash
   # Windows
   .\venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Start the Backend Server**:
   ```bash
   cd main_project
   python app.py
   ```
   
   You should see:
   ```
   * Running on http://0.0.0.0:5000
   * Debug mode: on
   ```

3. **Open the Web Interface**:
   - **Option 1**: Double-click `index.html`
   - **Option 2**: Open browser and navigate to: `file:///path/to/Project_OBD/index.html`

### **Using the System**

1. **Enter Engine Parameters**:
   - Engine RPM (e.g., 2200)
   - Lub Oil Pressure (e.g., 3.2)
   - Fuel Pressure (e.g., 45.0)
   - Coolant Pressure (e.g., 1.1)
   - Lub Oil Temperature (e.g., 85.0°C)
   - Coolant Temperature (e.g., 90.0°C)

2. **Click "🔬 Analyze Engine Status"**

3. **View Results**:
   - ✅ **Engine Status: Optimal** - Continue regular maintenance
   - ⚠️ **Engine Status: Requires Attention** - Schedule inspection

### **Theme Switching**
Click the **"🌙 Dark / ☀️ Light"** button to toggle between themes.

## 📈 Features

### **🎯 Core Functionality**
- **Real-time Engine Analysis** - Instant diagnostic results
- **Intelligent Predictions** - ML-powered condition assessment
- **Rule-based Safety Net** - Catches critical conditions
- **Professional Interface** - Clean, modern UI design
- **Theme Support** - Dark and light mode options

### **🔧 Technical Features**
- **RESTful API** - Clean backend architecture
- **Feature Engineering** - Advanced data preprocessing
- **Model Persistence** - Trained models saved for reuse
- **Cross-platform** - Works on Windows, macOS, Linux
- **Responsive Design** - Mobile and desktop compatible

### **📊 Data Analysis**
- **6 Primary Sensors** - Comprehensive engine monitoring
- **3 Engineered Features** - Enhanced prediction accuracy
- **Historical Analysis** - Based on 23,000+ data points
- **Threshold Detection** - Industry-standard warning levels

---

