from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load the trained model (Assuming you have saved the model as model.pkl)
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Preprocessing function
def preprocess_data(data):
    # Replace 'No internet service' and 'No phone service' with 'No'
    tobe_cleaned_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    data[tobe_cleaned_cols] = data[tobe_cleaned_cols].replace('No internet service', 'No')
    data['MultipleLines'] = data['MultipleLines'].replace('No phone service', 'No')
    
    # Convert TotalCharges to numeric
    data['TotalCharges'] = data['TotalCharges'].replace(" ", 0)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
    
    # Encode categorical features
    label_encoders = {}
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form.to_dict(flat=True)
    data = pd.DataFrame([data])
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Make prediction
    prediction = model.predict(data)
    
    # Return result
    result = 'Churn' if prediction[0] == 1 else 'No Churn'
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
