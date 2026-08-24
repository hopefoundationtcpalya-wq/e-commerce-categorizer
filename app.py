from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__ame__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '') or request.form.get('product_name', '') or request.form.get('description', '') or request.form.get('product', '')
        if not text and request.form:
            text = list(request.form.values())[0]

    if not text:
        return render_template('index.html', prediction="Please enter product description", prediction_text="Please enter product description", input_text="")

    prediction = model.predict([text])[0]

    if request.is_json:
        return jsonify({'category': prediction})
    else:
        msg = f"Predicted Category: {prediction}"
        return render_template('index.html', prediction=msg, prediction_text=msg, input_text=text)

if __name__ == '__main__':
    app.run(debug=True)
