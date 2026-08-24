from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Accept both JSON and form data
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '') or request.form.get('description', '') or request.form.get('product', '')

    if not text:
        # Try to get first form value if keys are different
        if request.form:
            text = list(request.form.values())[0]

    if not text:
        return render_template('index.html', prediction_text="Please enter product description", input_text="")

    prediction = model.predict([text])[0]

    # If request is JSON, return JSON, else return HTML page
    if request.is_json:
        return jsonify({'category': prediction})
    else:
        return render_template('index.html', prediction_text=f"Predicted Category: {prediction}", input_text=text)
