import sys
import unittest

sys.path.append('/Users/yuvalmoscovitz/Code/SentimentAnalysisProject')

#local modules
from src.data_preprocessing import split_sentence_and_label, clean_sentence, convert_label_to_numeric

class TestDataPreprocessing(unittest.TestCase):

    def test_clean_sentence(self):
        print("\n----------------- Starting test_clean_sentence Test ------------------")
        
        # Test based on the dataset
        dataset_sentences = [
            "According to Gran, the company has no plans to move all production to Russia, although that is where the company is growing .@neutral",
            "For the last quarter of 2010, Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier, while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m. @positive",
            "In the third quarter of 2010, net sales increased by 5.2% to EUR 205.5 mn, and operating profit by 34.9% to EUR 23.5 mn. @positive",
        ]
        
        for dataset_sentence in dataset_sentences:
            sentence, label = split_sentence_and_label(dataset_sentence)
            cleaned_sentence = clean_sentence(sentence)
            numeric_label = convert_label_to_numeric(label)
            
            print("Original Sentence:", sentence)
            print("Cleaned Sentence:", cleaned_sentence)
            print("Label:", label)
            print("Numeric Label:", numeric_label)
            print()
            
        
if __name__ == '__main__':
    unittest.main()
