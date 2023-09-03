from flask import Flask, request, jsonify
import requests
import os
import logging
from dotenv import load_dotenv
from src.data_loader import load_model
from src.model_predict import predict_sentiment_alpha_vantage
from src.sentiment_statistics import generate_sentiment_statistics
from src.utils import convert_numpy_int64

# Load the trained model
model = load_model('models/sentiment_classifier.pkl')

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

def fetch_news_from_alpha_vantage(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=10&apikey={api_key}'
    response = requests.get(url)
    return response.json()

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():
    try:
        stock_ticker = request.json['stock_ticker']
        logger.info(f"Received request for stock_ticker: {stock_ticker}")

        news_data = fetch_news_from_alpha_vantage(stock_ticker, ALPHA_VANTAGE_API_KEY)
        
        sentiment_analysis_results = predict_sentiment_alpha_vantage(news_data)
        
        statistics = generate_sentiment_statistics(sentiment_analysis_results)
        
        statistics = convert_numpy_int64(statistics)

        return jsonify({
            "stock_ticker": stock_ticker,
            "statistics": statistics
        })
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting the Flask server")
    app.run(debug=True)
