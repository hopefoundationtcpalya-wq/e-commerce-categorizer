from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'})

    prediction = model.predict([text])[0]
    return jsonify({'category': prediction})

if __name__ == '__main__':
    app.run(debug=True)
