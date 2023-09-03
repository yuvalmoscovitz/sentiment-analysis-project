import sys

import logging




from src.data_preprocessing import clean_sentence
from src.feature_extraction import sentence_to_vec
from src.data_loader import load_model, load_word2vec_model, load_top_n_indices

def predict_sentiment(input_news: str, model, word2vec_model, top_n_indices) -> int:
    """Predict the sentiment of a given news title."""
    cleaned_news = clean_sentence(input_news)
    feature_vector = sentence_to_vec(cleaned_news, word2vec_model)
    feature_vector_filtered = feature_vector[top_n_indices]
    
    prediction = model.predict([feature_vector_filtered])
    
    return prediction[0]

def predict_sentiment_alpha_vantage(alpha_vantage_news: dict) -> list:
    """Predict sentiments for multiple news titles from Alpha Vantage."""
    model_path = "models/sentiment_classifier.pkl"
    word2vec_model_path = "models/word2vec_model" 
    top_n_indices_path = "models/top_n_indices.pkl"  
    
    model = load_model(model_path)
    word2vec_model = load_word2vec_model(word2vec_model_path)
    top_n_indices = load_top_n_indices(top_n_indices_path)
    
    alpha_vantage_titles = extract_title_from_alpha_vantage_news(alpha_vantage_news)
    sentiments = []
    
    for news_dict in alpha_vantage_titles:
        news_title = news_dict['headline']
        title_sentiment = predict_sentiment(news_title, model, word2vec_model, top_n_indices)
        sentiments.append(title_sentiment)
        
    return sentiments


def extract_title_from_alpha_vantage_news(news_data: dict) -> list:
    """Extract news titles from Alpha Vantage news data."""
    headlines_and_sentiments = []
    for news_item in news_data['feed']:
        headline = news_item['title']
        headlines_and_sentiments.append({
            'headline': headline,
        })
    return headlines_and_sentiments

def sentiment_from_prediction(prediction: int) -> str:
    """Convert numerical prediction to human-readable sentiment."""
    if prediction == 1:
        return "The sentiment is Positive."
    elif prediction == 0:
        return "The sentiment is Neutral."
    else:
        return "The sentiment is Negative."
