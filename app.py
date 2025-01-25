from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
import numpy as np
import joblib  # Import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow cross-origin requests from any domain

# Define LSTM model architecture
def build_lstm_model(vocab_size, embedding_dim, max_len, output_classes):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_classes, activation='softmax'))
    return model

# Initialize model architecture
vocab_size = 8000  # Adjust according to your tokenizer vocabulary size
embedding_dim = 128
max_len = 100  # As per your padding
output_classes = 3  # Assuming 3 classes for sentiment: negative, neutral, positive

model = build_lstm_model(vocab_size, embedding_dim, max_len, output_classes)

# Load model weights
weights_path = "model_weights.keras"  # Replace with your weights file path
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at {weights_path}")

model.load_weights(weights_path)

# Load tokenizer
tokenizer_path = "tokenizer.joblib"  # Use joblib format
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

tokenizer = joblib.load(tokenizer_path)  # Load using joblib

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
