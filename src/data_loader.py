import csv
import pickle
import logging
from typing import Tuple, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def load_raw_data(file_paths: List[str], encoding: str = 'iso-8859-1') -> List[str]:
    all_lines = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                all_lines.extend(file.readlines())
        except FileNotFoundError:
            logger.error(f"File not found at {file_path}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing the file at {file_path}: {e}")
    return all_lines

def load_cleaned_data_csv(file_path: str, has_header: bool = True) -> Tuple[List[str], List[Union[int, str]]]:
    sentences = []
    labels = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            if has_header:
                next(reader)  
            for row in reader:
                sentence, label = row
                sentences.append(sentence)
                try:
                    labels.append(int(label)) 
                except ValueError:
                    labels.append(label)  
        return sentences, labels
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        return [], []
    except csv.Error:
        logger.error(f"Error reading CSV file at {file_path}")
        return [], []
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing the file at {file_path}: {e}")
        return [], []

def load_cleaned_data_pickled(file_path: str) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")
        return None, None
    except pickle.UnpicklingError:
        logger.error(f"Error unpickling file at {file_path}")
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing the file at {file_path}: {e}")
        return None, None

    
def load_testing_data(file_path: str) -> Tuple[Optional[List[str]], Optional[List[int]]]:
    """Load testing data and return it as (sentences, labels). Return (None, None) if loading fails."""
    sentences, labels = load_cleaned_data_csv(file_path)
    if sentences is None or labels is None:
        logger.error("Testing data file not found or could not be loaded.")
        return None, None
    return sentences, labels


def load_model(model_path):
    """Load the model from a file and return it. Return None if loading fails."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        logger.error("Model file not found.")
        return None
    
def load_word2vec_model(word2vec_model_path):
    try:
        from gensim.models import Word2Vec
        model = Word2Vec.load(word2vec_model_path)
        return model
    except FileNotFoundError:
        logger.error("Word2Vec model file not found.")
        return None
    
def load_top_n_indices(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
