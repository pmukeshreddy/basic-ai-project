"""
Feature Extraction Module
========================
Functions for extracting and transforming features from movie data.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

# Third-party dependencies (must be installed)
try:
    from textblob import TextBlob
except ImportError:
    print("Warning: TextBlob is not installed. Text sentiment analysis will not work.")
    print("Install with: pip install textblob")
    
    # Define a fallback class
    class TextBlob:
        def __init__(self, text):
            self.text = text
            
        @property
        def sentiment(self):
            class Sentiment:
                polarity = 0
                subjectivity = 0
            return Sentiment()

def extract_title_sentiment(df):
    """
    Extract sentiment scores from movie titles
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'title' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added title sentiment feature
    """
    df_features = df.copy()
    
    # Title sentiment
    df_features['title_sentiment'] = df_features['title'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) else 0
    )
    
    return df_features

def extract_sequel_indicator(df):
    """
    Detect if a movie title suggests it's a sequel
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'title' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added sequel indicator feature
    """
    df_features = df.copy()
    
    # Patterns indicating a sequel
    sequel_patterns = [
        r'\s+2\s*$', r'\s+3\s*$', r'\s+4\s*$', r'\s+5\s*$',  # Arabic numerals
        r'\s+ii\s*$', r'\s+iii\s*$', r'\s+iv\s*$', r'\s+v\s*$',  # Roman numerals
        r'\s+part\s+\d+', r'\s+chapter\s+\d+',  # Part or chapter indicators
        r'sequel', r'trilogy', r'the\s+next\s+chapter',  # Explicit sequel words
        r'returns', r'strikes\s+back', r'revenge', r'resurrection',  # Common sequel themes
        r'reloaded', r'continues', r'the\s+\w+\s+awakens'  # More sequel indicators
    ]

    def has_sequel_indicator(title):
        if not isinstance(title, str):
            return 0
        title_lower = title.lower()
        for pattern in sequel_patterns:
            if re.search(pattern, title_lower):
                return 1
        return 0

    df_features['contains_sequel'] = df_features['title'].apply(has_sequel_indicator)
    
    return df_features

def extract_keyword_presence(df):
    """
    Count occurrences of genre-specific keywords in movie overviews
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'overview' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added keyword presence features
    """
    df_features = df.copy()
    
    # Check if 'overview' column exists
    if 'overview' not in df_features.columns:
        print("Warning: 'overview' column not found. Skipping keyword extraction.")
        return df_features
    
    # Define keywords for different content types
    action_keywords = ['action', 'fight', 'battle', 'war', 'mission', 'explosion', 'danger',
                       'chase', 'combat', 'hero', 'terrorist', 'attack', 'weapon', 'survive']

    comedy_keywords = ['comedy', 'funny', 'laugh', 'humor', 'joke', 'hilarious', 'comic',
                       'amusing', 'silly', 'ridiculous', 'prank', 'comical']

    drama_keywords = ['drama', 'emotional', 'struggle', 'relationship', 'conflict', 'crisis',
                      'tragedy', 'suffering', 'dilemma', 'tension', 'turmoil']

    scifi_keywords = ['sci-fi', 'science fiction', 'future', 'space', 'alien', 'robot', 'technology',
                      'dystopian', 'apocalypse', 'advanced', 'cybernetic', 'galactic']

    horror_keywords = ['horror', 'scary', 'terrifying', 'fear', 'nightmare', 'monster', 'ghost',
                       'supernatural', 'haunt', 'demon', 'terrified', 'creature']

    def count_keywords(text, keyword_list):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        count = sum(1 for keyword in keyword_list if keyword in text_lower)
        return count

    df_features['action_keywords'] = df_features['overview'].apply(
        lambda x: count_keywords(x, action_keywords)
    )

    df_features['comedy_keywords'] = df_features['overview'].apply(
        lambda x: count_keywords(x, comedy_keywords)
    )

    df_features['drama_keywords'] = df_features['overview'].apply(
        lambda x: count_keywords(x, drama_keywords)
    )

    df_features['scifi_keywords'] = df_features['overview'].apply(
        lambda x: count_keywords(x, scifi_keywords)
    )

    df_features['horror_keywords'] = df_features['overview'].apply(
        lambda x: count_keywords(x, horror_keywords)
    )
    
    return df_features

def extract_overview_sentiment(df):
    """
    Extract sentiment and subjectivity scores from movie overviews
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'overview' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added overview sentiment features
    """
    df_features = df.copy()
    
    # Check if 'overview' column exists
    if 'overview' not in df_features.columns:
        print("Warning: 'overview' column not found. Skipping overview sentiment extraction.")
        return df_features
    
    # Overview sentiment
    df_features['overview_sentiment'] = df_features['overview'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) else 0
    )

    # Overview subjectivity
    df_features['overview_subjectivity'] = df_features['overview'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if isinstance(x, str) else 0
    )
    
    return df_features

def extract_release_season(df):
    """
    Extract season information from movie release dates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'release_date' column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added release season features
    """
    df_features = df.copy()
    
    # Check if 'release_date' column exists
    if 'release_date' not in df_features.columns:
        print("Warning: 'release_date' column not found. Skipping season extraction.")
        return df_features
    
    def get_season(date_str):
        if not isinstance(date_str, str):
            return 'unknown'

        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            month = date_obj.month

            if 3 <= month <= 5:
                return 'spring'
            elif 6 <= month <= 8:
                return 'summer'
            elif 9 <= month <= 11:
                return 'fall'
            else:
                return 'winter'
        except:
            return 'unknown'

    df_features['season'] = df_features['release_date'].apply(get_season)

    # Convert season to one-hot encoding
    season_dummies = pd.get_dummies(df_features['season'], prefix='season')
    df_features = pd.concat([df_features, season_dummies], axis=1)
    
    return df_features

def extract_features(data):
    """
    Extract all features for genre prediction
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with necessary columns
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with all extracted features
    """
    # Create a copy to avoid modifying the original
    df_features = data.copy()
    
    # 1. Title sentiment
    df_features = extract_title_sentiment(df_features)
    
    # 2. Contains sequel indicator
    df_features = extract_sequel_indicator(df_features)
    
    # 3. Keyword presence in overview
    df_features = extract_keyword_presence(df_features)
    
    # 4. Sentiment analysis of overview
    df_features = extract_overview_sentiment(df_features)
    
    # 5. Season from release date
    df_features = extract_release_season(df_features)
    
    print(f"Extracted {df_features.shape[1] - data.shape[1]} new features")
    return df_features

# Additional text preprocessing functionality

def preprocess_text(df, text_columns=['title', 'overview']):
    """
    Preprocess text columns: tokenize, remove stopwords, etc.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_columns : list
        List of text columns to process
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with processed text columns
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        
        # Try to find NLTK resources, download if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        
    except ImportError:
        print("Warning: NLTK is not installed. Using simplified text processing.")
        print("Install with: pip install nltk")
        
        # Simplified processing without NLTK
        stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ])
        
        # Simple stemming function
        def simple_stem(word):
            # Very basic stemming rules
            if word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ed'):
                return word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                return word[:-1]
            return word
        
        stemmer = type('SimplePorterStemmer', (), {'stem': staticmethod(simple_stem)})
    
    df_processed = df.copy()
    
    def process_text(text):
        if not isinstance(text, str) or not text:
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Simple tokenization
        tokens = text.split()

        # Remove stop words and stem
        processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

        return ' '.join(processed_tokens)

    # Apply text processing to each column
    for column in text_columns:
        if column in df_processed.columns:
            df_processed[f'{column}_processed'] = df_processed[column].apply(process_text)

    print(f"Text preprocessing completed for columns: {text_columns}")
    return df_processed
