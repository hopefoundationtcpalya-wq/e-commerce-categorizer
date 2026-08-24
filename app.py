from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        product_name = request.form['product']
        product_vec = vectorizer.transform([product_name])
        prediction = model.predict(product_vec)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '_main_':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
