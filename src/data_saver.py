import csv
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_data_to_csv(data, file_path, header=None):
    """Save data to a CSV file.

    Parameters:
    - data (List[Tuple]): The data to be saved.
    - file_path (str): The path where the CSV file will be saved.
    - header (List[str], optional): The header row for the CSV file.
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if header:
            writer.writerow(header)
        writer.writerows(data)
    logger.info(f"Data saved to: {file_path}")

def save_cleaned_data_csv(sentences: list, labels: list, output_file: str):
    """Save cleaned data to a CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'Label'])
        writer.writerows(zip(sentences, labels))
    logger.info(f"Cleaned data saved to {output_file}")

def save_cleaned_data_pickle(sentences: list, labels: list, output_file: str):
    """Save cleaned data to a Pickle file."""
    with open(output_file, 'wb') as f:
        pickle.dump(list(zip(sentences, labels)), f)
    logger.info(f"Cleaned data saved to {output_file}")

def csv_to_pickle(csv_file_path, pickle_file_path):
    try:
        # Read data from CSV
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader]
        
        # Save data to Pickle
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Successfully converted {csv_file_path} to {pickle_file_path}")
        
    except FileNotFoundError:
        logger.error(f"File not found at {csv_file_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
