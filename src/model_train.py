import pickle
import logging
import numpy as np
import pandas as pd
from data_loader import load_cleaned_data_csv  
from feature_extraction import train_word2vec, sentence_to_vec  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def train_and_save_model(sentences, labels, model_output_path, train_size=0.8):
    """
    Train a Random Forest classifier, extract feature importances, 
    retrain the model with top N features, and save the trained model.
    
    Parameters:
    - sentences (list): List of sentences to train on.
    - labels (list): List of labels corresponding to the sentences.
    - model_output_path (str): File path to save the trained model.
    - train_size (float, optional): Proportion of data to use for training. Default is 0.8.
    """
    # Train the Word2Vec model
    tokenized_sentences = [sentence.split() for sentence in sentences]
    word2vec_model = train_word2vec(tokenized_sentences)
    
    # Convert sentences to vectors
    sentence_vectors = np.array([sentence_to_vec(sentence, word2vec_model) for sentence in sentences])
    
    # Split the dataset into training and testing sets
    train_size = int(train_size * len(sentence_vectors))
    train_set, test_set = sentence_vectors[:train_size], sentence_vectors[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]
        
    # Train the Random Forest classifier
    classifier = RandomForestClassifier()  # Initialize Random Forest
    classifier.fit(train_set, train_labels)  # Fit the model
    
    # Extract feature importances
    feature_importances = classifier.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': [f'feature_{i}' for i in range(len(feature_importances))],
        'Importance': feature_importances
    })

    # Sort the DataFrame by the importances
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    # Select top N important features
    top_n = 7
    top_n_indices = feature_importance_df.index[:top_n].tolist()

    # Filter your training and testing sets to include only the top N features
    train_set_filtered = train_set[:, top_n_indices]
    test_set_filtered = test_set[:, top_n_indices]

    # Retrain your model using only the top N features
    classifier.fit(train_set_filtered, train_labels)

    # Save the trained model using pickle
    with open(model_output_path, 'wb') as model_file:
        pickle.dump(classifier, model_file)
    
    # Combine test_set and test_labels for saving
    test_data = np.column_stack((test_set_filtered, test_labels))
    
    # Create a DataFrame and save to CSV
    columns = [f'feature_{i}' for i in range(test_set_filtered.shape[1])] + ['label']
    test_data_df = pd.DataFrame(test_data, columns=columns)
    test_data_df.to_csv("data/processed_data/testing_data.csv", index=False)
    
    logger.info("Trained model saved to: %s", model_output_path)
    
    return classifier

def main():
    logger.info("Start training the model")
    cleaned_data_file = "data/processed_data/cleaned_data.csv"
    data = load_cleaned_data_csv(cleaned_data_file)
    
    if data is None:
        logging.error("Failed to load data. Exiting.")
        return

    # Assuming your data is a tuple of (sentences, labels)
    sentences, labels = data
    
    # Train and save the model
    model_output_path = "models/sentiment_classifier.pkl"
    train_and_save_model(sentences, labels, model_output_path)
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
