# Intelligent Chatbot for KU Website

This repository contains the implementation of an intelligent chatbot designed to enhance user experience on the KU university website. It provides instant, automated responses to common queries, leveraging Natural Language Processing (NLP) and a Bidirectional Long Short-Term Memory (BiLSTM) model.

## Features

- **Accurate Intent Recognition**: Uses a BiLSTM model to understand user queries with high precision.
- **Responsive Web Interface**: A user-friendly interface built with HTML, CSS, and JavaScript.
- **Real-time Interaction**: Flask backend ensures seamless communication between the chatbot and users.
- **University-Specific Query Handling**: Supports 19 tags tailored to common university queries like academics, research, athletics, and more.

## Technologies Used

- **Backend**: Flask
- **Machine Learning**: TensorFlow, BiLSTM model
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NLTK, NumPy, pandas
- **Deployment**: Hosted on a cloud platform for accessibility

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ku-chatbot.git
   cd ku-chatbot
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install additional dependencies:
   ```bash
   pip install tensorflow nltk
   ```

4. Download the `intents.json` dataset and place it in the `data` directory.

## Usage

### Training the Model

To train the chatbot model:
1. Run the training script:
   ```bash
   python train_chatbot.py
   ```
2. The trained model, tokenizer, and tag encoder will be saved in the `model` directory.

### Starting the Server

To start the Flask server:
```bash
python app.py
```
Visit `http://localhost:5000` in your browser to interact with the chatbot.

### Customizing Responses

Modify the `intents.json` file to add or update chatbot responses. Example format:
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Good morning"],
      "responses": ["Hello!", "Hi there! How can I help you?"]
    }
  ]
}
```

## Project Structure

```
ku-chatbot/
├── data/
│   └── intents.json         # Dataset with intents, patterns, and responses
├── model/
│   ├── chatbot_model.h5     # Trained BiLSTM model
│   ├── tag_encoder.pkl      # Label encoder for tags
│   └── vectorizer.pkl       # Tokenizer for word embeddings
├── static/
│   └── styles.css           # Styling for the web interface
├── templates/
│   └── index.html           # Chatbot frontend
├── app.py                   # Flask server code
├── train_chatbot.py         # Model training script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Future Improvements

- Integration with advanced models like BERT for better contextual understanding.
- Support for voice-to-text interaction.
- Real-time updates for campus events and notifications.

**Author**: Jagadeesh Sai Dokku  
**Date of Completion**: December 2024
```
