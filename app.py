from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as file:
        model, vectorizer = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not available'}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No valid text provided'}), 400

    try:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        return jsonify({'sentiment': prediction})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
