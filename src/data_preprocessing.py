import re
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_sentence_and_label(line: str) -> tuple:
    """Split a line into sentence and label."""
    sentence, label = line.split('@')
    return sentence.strip(), label.strip()

def remove_stopwords(tokens: list) -> list:
    """Remove stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]

def apply_stemming(tokens: list) -> list:
    """Apply stemming to a list of tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def clean_sentence(sentence: str) -> str:
    """Clean and tokenize a sentence."""
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z0-9\s%!?,.]', '', sentence)
    tokens = word_tokenize(sentence)
    #tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    return ' '.join(tokens)

def convert_label_to_numeric(label: str) -> int:
    """Convert textual label to numeric."""
    label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    return label_mapping.get(label, 0)

def preprocess_data(lines: list) -> tuple:
    """Preprocess a list of lines to extract sentences and labels."""
    sentences, labels = [], []
    for line in lines:
        sentence, label = split_sentence_and_label(line)
        sentences.append(clean_sentence(sentence))
        labels.append(convert_label_to_numeric(label))
    return sentences, labels





