from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained models and preprocessing objects
ensemble_model = joblib.load('./models/voting_classifier_model.pkl')  # Adjust the path as needed
scaler = joblib.load('./models/scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        n_value = float(request.form['N'])
        p_value = float(request.form['P'])
        k_value = float(request.form['K'])
        temperature_value = float(request.form['temperature'])
        humidity_value = float(request.form['humidity'])
        ph_value = float(request.form['ph'])
        rainfall_value = float(request.form['rainfall'])

        # Prepare the input data
        user_data = pd.DataFrame({
            'N': [n_value],
            'P': [p_value],
            'K': [k_value],
            'temperature': [temperature_value],
            'humidity': [humidity_value],
            'ph': [ph_value],
            'rainfall': [rainfall_value]
        })

        # Normalize the data
        user_data_normalized = scaler.transform(user_data)

        # Make predictions using the ensemble model
        prediction_ensemble = ensemble_model.predict(user_data_normalized)[0]

        return render_template('result.html', label_ensemble=prediction_ensemble)

if __name__ == '__main__':
    app.run(debug=True)
