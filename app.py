import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

def ValuePredictor(new_l):
    # Reshape input data into 2D array
    to_predict = np.array(new_l).reshape(1, -1)
    # Load the model
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    # Predict the result
    result = loaded_model.predict(to_predict)
    return result[0]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get form data and preprocess it
        to_predict_list = [1 if x == 'male' else 0 if x == 'female' else 1 if x == 'yes' else 0 if x == 'no' else x for x in request.form.values()]
        # Convert to float
        new_l = list(map(float, to_predict_list))
        # Make prediction
        result = ValuePredictor(new_l)
        # Interpret the prediction
        if int(result) == 1:
            prediction = 'Yes, person can claim insurance.'
        else:
            prediction = 'No, person will not claim insurance.'
        return render_template("result.html", prediction=prediction)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
