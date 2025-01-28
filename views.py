from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from .nlp_utils import (
    load_sentiment_dataset,
    load_dataset,
    classify_sentiment_bert,  # Updated function for BERT-based sentiment analysis
    find_sentiment_from_dataset,
    detect_greeting,
    time_based_greeting
)
import json
import random

# Load datasets
faq_dataset = load_dataset('chatbot/data/Cleaned_Mental_Health_FAQ.csv')  # Load FAQ dataset
faq_dict = {entry['Questions'].strip().lower(): entry['Answers'].strip() for entry in faq_dataset}  # Build FAQ dictionary
sentiment_dataset = load_sentiment_dataset('chatbot/data/cleaned_sentiment_dataset.csv')  

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()

            if not user_message:
                return JsonResponse({'response': "It seems like you didn't say anything. Could you try again?"})

            # Debugging: Log user message and FAQ questions
            print(f"User message: {user_message}")
            print(f"Loaded FAQ dataset questions: {list(faq_dict.keys())}")

            # Check for greetings
            if detect_greeting(user_message):
                greeting_responses = [
                    "Hello! How can I assist you today?",
                    "Hi there! What’s on your mind?",
                    "Hey! I’m here to help. How can I support you?",
                    time_based_greeting()
                ]
                response_message = random.choice(greeting_responses)
                return JsonResponse({'response': response_message})

            # FAQ matching logic
            faq_response = None
            for question, answer in faq_dict.items():
                if user_message.lower() in question:
                    faq_response = answer
                    break

            if faq_response:
                return JsonResponse({'response': faq_response})

            # Classify sentiment using BERT (replaces TextBlob)
            sentiment = classify_sentiment_bert(user_message)

            # Fallback to dataset-based sentiment matching if needed
            if not sentiment:
                sentiment = find_sentiment_from_dataset(user_message, sentiment_dataset)

            # Generate response based on sentiment
            sentiment_responses = {
                'positive': [
                    "That's great to hear! Tell me more.",
                    "I'm glad you're feeling positive!",
                    "Wonderful! Keep the good vibes going."
                ],
                'negative': [
                    "I'm sorry to hear that. Want to talk about it?",
                    "That sounds tough. I'm here to listen.",
                    "Take your time. It's okay to feel this way."
                ],
                'neutral': [
                    "I see. Is there more you'd like to share?",
                    "Thanks for sharing. Feel free to continue.",
                    "I'm here to chat whenever you're ready."
                ]
            }

            response_message = random.choice(sentiment_responses.get(sentiment, ["I'm here to listen. Please share more."]))

            return JsonResponse({'response': response_message})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def index(request):
    return JsonResponse({"message": "Welcome to the Mental Health Chatbot!"})

def chatbot_page(request):
    return render(request, 'chatbot/index.html')
