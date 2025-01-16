from flask import Flask, render_template, request, jsonify
import random
import json
import numpy as np
import pickle
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize the Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model = tf.keras.models.load_model('model/chatbot_model.h5')

# Load the tokenizer
with open('vectorizer.pkl', 'rb') as f:
    tokenizer_word_index = pickle.load(f)  # Load the word index

# Recreate the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.word_index = tokenizer_word_index  # Assign the loaded word index to tokenizer

# Load the tag encoder
with open('tag_encoder.pkl', 'rb') as f:
    tag_encoder = pickle.load(f)

# Load intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_input(text):
    tokens = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    return ' '.join(lemmatized_words)

def pad_sequences(sequences, maxlen):
    # Create an array of zeros
    padded_sequences = np.zeros((len(sequences), maxlen))
    
    for i, seq in enumerate(sequences):
        # Pad sequences on the right
        padded_sequences[i, :len(seq)] = seq[:maxlen]
        
    return padded_sequences

def get_response(msg):
    try:
        processed_msg = preprocess_input(msg)
        X = tokenizer.texts_to_sequences([processed_msg])  # Use tokenizer to convert text to sequences
        
        # Set maxlen to a known value or calculate based on your training data.
        maxlen = 10  # Replace with the actual maxlen you used during training
        
        X_padded = pad_sequences(X, maxlen)
        X_reshaped = X_padded.reshape(X_padded.shape[0], X_padded.shape[1], 1)

        prediction = model.predict(X_reshaped)
        predicted_class = np.argmax(prediction)
        tag = tag_encoder.inverse_transform([predicted_class])[0]
        prob = prediction[0][predicted_class]

        print(f"Prediction: {prediction}, Predicted class: {predicted_class}, Tag: {tag}, Probability: {prob}")

        if prob > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        elif prob > 0.5:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return f"I'm not completely sure, but I think: {random.choice(intent['responses'])}"
        
        return "I do not understand..."
    except Exception as e:
        return f"Error occurred: {str(e)}"


@app.route("/", defaults={"path": ""})
def serve_react_app(path):
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_text = request.form["message"]
    response = get_response(user_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
