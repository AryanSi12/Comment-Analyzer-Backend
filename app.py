from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib  # Import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow cross-origin requests from any domain

# Load trained model
model_path = "my_model.keras"  # Replace with your model's folder path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Load tokenizer
tokenizer_path = "tokenizer.joblib"  # Use joblib format
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

tokenizer = joblib.load(tokenizer_path)  # Load using joblib

# Tokenizer parameters
max_len = 100

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get comments from request
        data = request.get_json()
        comments = data.get('comments', [])

        if not comments or not isinstance(comments, list):
            return jsonify({'error': 'Invalid input. Provide a list of comments.'}), 400

        # Preprocess comments
        sequences = tokenizer.texts_to_sequences(comments)
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

        # Predict sentiment
        predictions = model.predict(padded_sequences)
        
        # Assuming the model outputs softmax (multi-class), change if needed
        sentiment_labels = ['negative', 'neutral', 'positive']  # Update based on your label encoding

        # Map predictions to sentiment labels 
        results = [
            {'comment': comment, 'sentiment': sentiment_labels[np.argmax(pred)]}
            for comment, pred in zip(comments, predictions)
        ] 

        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
