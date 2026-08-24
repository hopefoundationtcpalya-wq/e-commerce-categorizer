from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(_name_)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Accept both JSON and form data
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '') or request.form.get('description', '') or request.form.get('product', '')
        if not text and request.form:
            text = list(request.form.values())[0]

    if not text:
        return render_template('index.html', prediction_text="Please enter product description", input_text="")

    prediction = model.predict([text])[0]

    if request.is_json:
        return jsonify({'category': prediction})
    else:
        return render_template('index.html', prediction_text=f"Predicted Category: {prediction}", input_text=text)

if _name_ == '_main_':
    app.run(debug=True)
