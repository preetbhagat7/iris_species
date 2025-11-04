from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict
        prediction = model.predict(features)[0]

        species = ['Setosa', 'Versicolor', 'Virginica']
        result = species[prediction]

        return render_template('index.html', prediction_text=f'Predicted Flower: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)

