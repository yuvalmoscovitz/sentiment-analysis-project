from gensim.models import Word2Vec
import numpy as np

def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    """
    Train a Word2Vec model.

    Parameters:
    - sentences: List of tokenized sentences.
    - vector_size: Size of word vectors.
    - window: Context window size.
    - min_count: Minimum word frequency.

    Returns:
    - Trained Word2Vec model.
    """
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count)
    model.build_vocab(sentences)  
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def sentence_to_vec(sentence, model):
    """
    Convert a sentence to a vector by averaging the vectors of the words in the sentence.

    Parameters:
    - sentence (str): The sentence to convert.
    - model (Word2Vec): Trained Word2Vec model.

    Returns:
    - np.array: Sentence vector.
    """
    words = sentence.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv.index_to_key]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    sentence_vec = np.mean(word_vecs, axis=0)
    return sentence_vec


