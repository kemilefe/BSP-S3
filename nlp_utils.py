import pandas as pd
from difflib import get_close_matches
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load BERT model and tokenizer
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = BertForSequenceClassification.from_pretrained('/Users/kemalettinefegedik/Desktop/bsp/sentiment_model')
tokenizer = BertTokenizer.from_pretrained("/Users/kemalettinefegedik/Desktop/bsp/sentiment_model")
model.to(device)
model.eval()

def load_dataset(filepath):
    """
    Loads the FAQ dataset from a CSV file.
    Expects columns: 'Questions', 'Answers'.
    """
    try:
        # Load the dataset
        data = pd.read_csv(filepath)

        # Ensure required columns exist
        if 'Questions' not in data.columns or 'Answers' not in data.columns:
            raise ValueError("Dataset must have 'Questions' and 'Answers' columns.")
        
        # Convert to a list of dictionaries for easy access
        return data.to_dict(orient='records')
    except Exception as e:
        print(f"Error loading FAQ dataset: {e}")
        return []

def load_sentiment_dataset(filepath):
    """
    Loads the sentiment analysis dataset from a CSV file.
    Expects columns: 'message', 'sentiment'.
    """
    try:
        # Load the dataset
        data = pd.read_csv(filepath)

        # Ensure required columns exist
        if 'message' not in data.columns or 'sentiment' not in data.columns:
            raise ValueError("Dataset must have 'message' and 'sentiment' columns.")
        
        # Convert to a list of dictionaries for easy access
        return data.to_dict(orient='records')
    except Exception as e:
        print(f"Error loading sentiment dataset: {e}")
        return []

def classify_sentiment_bert(message):
    """
    Classifies sentiment as positive, negative, or neutral using BERT.
    """
    inputs = tokenizer(message, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        sentiment_id = torch.argmax(logits, dim=1).item()

    # Map sentiment IDs to labels
    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_labels.get(sentiment_id, "neutral")

def find_sentiment_from_dataset(user_message, dataset):
    """
    Matches the user's message with the closest message in the dataset to determine sentiment.
    """
    messages = [entry['message'] for entry in dataset]
    closest_match = get_close_matches(user_message.lower(), messages, n=1, cutoff=0.5)
    if closest_match:
        for entry in dataset:
            if entry['message'].lower() == closest_match[0]:
                return entry['sentiment']
    return 'neutral'

def detect_greeting(message):
    """
    Detects if a message is a greeting.
    """
    greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'greetings']
    message = message.lower()

    # Check if any greeting keyword is in the message
    return any(word in message for word in greeting_keywords)

def time_based_greeting():
    """
    Generates a greeting based on the time of day.
    """
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning! How can I help you today?"
    elif current_hour < 18:
        return "Good afternoon! How can I assist you?"
    else:
        return "Good evening! What’s on your mind?"
