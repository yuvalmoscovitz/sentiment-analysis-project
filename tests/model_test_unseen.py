import sys
import os
import logging

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_loader import load_model, load_word2vec_model, load_top_n_indices
from src.model_predict import predict_sentiment, sentiment_from_prediction

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_input():
    return input("Enter the market news for sentiment analysis (type 'exit' to quit): ")

def main():
    model_path = "models/sentiment_classifier.pkl"
    word2vec_model_path = "models/word2vec_model" 
    top_n_indices_path = "models/top_n_indices.pkl"  
    
    model = load_model(model_path)
    word2vec_model = load_word2vec_model(word2vec_model_path)
    top_n_indices = load_top_n_indices(top_n_indices_path)
    
    if model is None or word2vec_model is None or top_n_indices is None:
        logger.error("Exiting due to missing model files.")
        return
    
    while True:
        input_news = get_user_input()
        
        if input_news.lower() == 'exit':
            break
        
        verdict = predict_sentiment(input_news, model, word2vec_model, top_n_indices)
        sentiment = sentiment_from_prediction(verdict)
        print(sentiment)

if __name__ == "__main__":
    main()
