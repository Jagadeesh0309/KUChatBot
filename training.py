import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from keras.utils.vis_utils import plot_model


# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Prepare the data
patterns = []
tags = []
responses = {}

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
X = tokenizer.texts_to_sequences(patterns)

# Determine the maximum length of sequences
maxlen = max(len(seq) for seq in X)

# Custom padding function
def pad_sequences(sequences, maxlen):
    # Create an array of zeros
    padded_sequences = np.zeros((len(sequences), maxlen))
    
    for i, seq in enumerate(sequences):
        # Pad sequences on the right
        padded_sequences[i, :len(seq)] = seq[:maxlen]
        
    return padded_sequences

# Pad the sequences
X = pad_sequences(X, maxlen)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# One-hot encode the labels
y = pd.get_dummies(y).values  # Convert to one-hot encoding

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Bidirectional LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))  # Dropout after embedding layer
model.add(Bidirectional(LSTM(100, return_sequences=True)))  # Return sequences for the next LSTM layer
model.add(Dropout(0.2))  # Dropout after LSTM layer
model.add(Bidirectional(LSTM(100)))  # Another Bidirectional LSTM layer
model.add(Dropout(0.2))  # Dropout after the second LSTM layer
model.add(Dense(len(responses), activation='softmax'))

# Compile the model with accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with validation data
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val))

# Save the trained model
model.save('model/chatbot_model.h5')
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Save the tokenizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tokenizer.word_index, f)  # Save the word index of the tokenizer

# Save the tag encoder
with open('tag_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Function to get a response
def get_response(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen)
    predicted = model.predict(padded)
    tag_index = np.argmax(predicted, axis=1)
    tag = label_encoder.inverse_transform(tag_index)  # Decode the tag back to original
    return np.random.choice(responses[tag[0]])

# Example usage with multiple queries
queries = [
    "Hi",
    "Hello",
    "Goodbye",
    "Where can international students find support at KU?",
    "Where can I find the KU registrar?",
    "Thanks for the help!",
    "See you later",
    "How do I find out my class times?",
    "How do I use KU Print Services?"
]

# Iterate over each query and print the response
for query in queries:
    response = get_response(query)
    print(f"User: {query}\nBot: {response}\n")

# Plot training & validation loss and accuracy
def plot_history(history):
    # Loss Plot
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_history(history)

