import pandas as pd
import logging
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_model, load_testing_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path, testing_data_path):
        """
        Initialize the ModelTester class.
        
        Parameters:
        - model_path (str): File path to the trained model.
        - testing_data_path (str): File path to the testing data.
        """
        self.model = load_model(model_path)
        self.testing_data_path = testing_data_path
        
    def evaluate_model(self):
        """
        Evaluate the trained model on the testing data.
        """
        if self.model is None:
            logger.error("Model is not loaded. Exiting.")
            return
        
        # Load the testing data from the CSV file
        test_data_df = pd.read_csv(self.testing_data_path)
        
        # Assuming the last column is 'label' and the rest are features
        test_set = test_data_df.iloc[:, :-1].values
        test_labels = test_data_df.iloc[:, -1].values
        
        # Make predictions
        predictions = self.model.predict(test_set)
        
        # Evaluate the model
        accuracy = accuracy_score(test_labels, predictions)
        logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        # Detailed classification report
        report = classification_report(test_labels, predictions)
        logger.info("Classification Report:")
        logger.info(report)

def main():
    logger.info("Start evaluating the model")
    
    model_path = "models/sentiment_classifier.pkl"
    testing_data_path = "data/processed_data/testing_data.csv"
    
    tester = ModelTester(model_path, testing_data_path)
    tester.evaluate_model()
    
    logger.info("Evaluation completed.")

if __name__ == "__main__":
    main()
