import numpy as np
from itertools import groupby
import statistics as py_statistics  
from collections import Counter

def calculate_basic_statistics(sentiments):
    sentiments_array = np.array(sentiments)
    total_news = sentiments_array.size
    positive_count = np.sum(sentiments_array == 1)
    neutral_count = np.sum(sentiments_array == 0)
    negative_count = np.sum(sentiments_array == -1)
    
    return {
        'Total News Articles': int(total_news),
        'Positive Sentiments': int(positive_count),
        'Neutral Sentiments': int(neutral_count),
        'Negative Sentiments': int(negative_count),
    }

def calculate_advanced_statistics(sentiments):
    sentiments_array = np.array(sentiments)
    most_common = Counter(sentiments).most_common(1)[0][0]
    
    return {
        'Positive Percentage': np.mean(sentiments_array == 1) * 100,
        'Neutral Percentage': np.mean(sentiments_array == 0) * 100,
        'Negative Percentage': np.mean(sentiments_array == -1) * 100,
        'Longest Positive Streak': calculate_longest_streak(sentiments, 1),
        'Longest Neutral Streak': calculate_longest_streak(sentiments, 0),
        'Longest Negative Streak': calculate_longest_streak(sentiments, -1),
        'Sentiment Transitions': calculate_transitions(sentiments),
        'Most Common in a Row': most_common,
        'Standard Deviation': np.std(sentiments_array),
        'Median Sentiment': np.median(sentiments_array),
        'Mode Sentiment': py_statistics.mode(sentiments)
    }

def calculate_longest_streak(sentiments, value):
    return max((sum(1 for _ in g) for k, g in groupby(sentiments) if k == value), default=0)

def calculate_transitions(sentiments):
    return sum(1 for a, b in zip(sentiments, sentiments[1:]) if a != b)

def generate_sentiment_statistics(sentiments):
    basic_stats = calculate_basic_statistics(sentiments)
    advanced_stats = calculate_advanced_statistics(sentiments)
    return {'Basic Statistics': basic_stats, 'Advanced Statistics': advanced_stats}
