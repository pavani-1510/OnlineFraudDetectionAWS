from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('/home/pavani-r/Documents/VSCODE/Online Fraud Detection/fraud_detection_model.pkl')
scaler = joblib.load('/home/pavani-r/Documents/VSCODE/Online Fraud Detection/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the form data from the user
            type_ = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])
            isFraud = int(request.form['isFraud'])

            # Create a DataFrame from the input
            input_data = pd.DataFrame([{
                'type': type_,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest,
                'isFraud': isFraud
            }])

            # Encode the 'type' column (if applicable)
            input_data['type'] = input_data['type'].apply(lambda x: 0 if x == 'payment' else 1)  # Example: 'payment' -> 0, 'transfer' -> 1

            # Scale the input data
            scaled_input = scaler.transform(input_data.drop('isFraud', axis=1))

            # Make a prediction
            prediction = model.predict(scaled_input)

            # Display the result
            result = f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}"
            return render_template('index.html', result=result)

        except Exception as e:
            result = f"Error in processing input: {e}"
            return render_template('index.html', result=result)

    return render_template('index.html', result='No data submitted.')

if __name__ == '__main__':
    app.run(debug=True)
