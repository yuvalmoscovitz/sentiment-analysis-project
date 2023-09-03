import os
import logging
from model_train import train_and_save_model
from data_loader import load_raw_data, load_cleaned_data_csv
from data_preprocessing import preprocess_data
from data_saver import save_cleaned_data_csv, save_cleaned_data_pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Main function started")
    cleaned_data_csv_output_file = "data/processed_data/cleaned_data.csv"
    cleaned_data_pickle_output_file = "data/processed_data/cleaned_data.pkl"
    
    raw_file_paths = [
        "data/raw_data/FinancialPhraseBank-v1.0 2/Sentences_AllAgree.txt",
        "data/raw_data/FinancialPhraseBank-v1.0 2/Sentences_75Agree.txt",
        "data/raw_data/FinancialPhraseBank-v1.0 2/Sentences_66Agree.txt"
    ]
    raw_lines = load_raw_data(raw_file_paths)

    sentences, labels = preprocess_data(raw_lines)

    processed_data_dir = os.path.join("data", "processed_data")
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    save_cleaned_data_csv(sentences, labels, cleaned_data_csv_output_file)
    save_cleaned_data_pickle(sentences, labels, cleaned_data_pickle_output_file)
    logger.info(f"Cleaned data saved to: {cleaned_data_csv_output_file} and {cleaned_data_pickle_output_file}")

    sentences, labels = load_cleaned_data_csv(cleaned_data_csv_output_file)
    
    model_output_path = "models/sentiment_classifier.pkl"
    word2vec_output_path = "models/word2vec_model"
    train_and_save_model(sentences, labels, model_output_path, word2vec_output_path)
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
