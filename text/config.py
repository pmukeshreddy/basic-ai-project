"""
Configuration Module
==================
Configuration settings for movie genre classification.
"""

# Mapping of genres to groups
GENRE_TO_GROUP = {
    # Group 1: Drama, Music
    'Drama': 'Group 1',
    'Music': 'Group 1',
    # Group 2: Comedy, Crime, History
    'Comedy': 'Group 2',
    'Crime': 'Group 2',
    'History': 'Group 2',
    # Group 3: Thriller, Science Fiction, Fantasy, Documentary, Western
    'Thriller': 'Group 3',
    'Science Fiction': 'Group 3',
    'Fantasy': 'Group 3',
    'Documentary': 'Group 3',
    'Western': 'Group 3',
    # Group 4: Action, Horror, Mystery, War, TV Movie
    'Action': 'Group 4',
    'Horror': 'Group 4',
    'Mystery': 'Group 4',
    'War': 'Group 4',
    'TV Movie': 'Group 4',
    # Group 5: Adventure, Romance, Family, Animation
    'Adventure': 'Group 5',
    'Romance': 'Group 5',
    'Family': 'Group 5',
    'Animation': 'Group 5'
}

# Feature extraction settings
SEQUEL_PATTERNS = [
    r'\s+2\s*$', r'\s+3\s*$', r'\s+4\s*$', r'\s+5\s*$',  # Arabic numerals
    r'\s+ii\s*$', r'\s+iii\s*$', r'\s+iv\s*$', r'\s+v\s*$',  # Roman numerals
    r'\s+part\s+\d+', r'\s+chapter\s+\d+',  # Part or chapter indicators
    r'sequel', r'trilogy', r'the\s+next\s+chapter',  # Explicit sequel words
    r'returns', r'strikes\s+back', r'revenge', r'resurrection',  # Common sequel themes
    r'reloaded', r'continues', r'the\s+\w+\s+awakens'  # More sequel indicators
]

# Keywords for genre detection
GENRE_KEYWORDS = {
    'action': ['action', 'fight', 'battle', 'war', 'mission', 'explosion', 'danger',
               'chase', 'combat', 'hero', 'terrorist', 'attack', 'weapon', 'survive'],
    
    'comedy': ['comedy', 'funny', 'laugh', 'humor', 'joke', 'hilarious', 'comic',
               'amusing', 'silly', 'ridiculous', 'prank', 'comical'],
    
    'drama': ['drama', 'emotional', 'struggle', 'relationship', 'conflict', 'crisis',
              'tragedy', 'suffering', 'dilemma', 'tension', 'turmoil'],
    
    'scifi': ['sci-fi', 'science fiction', 'future', 'space', 'alien', 'robot', 'technology',
              'dystopian', 'apocalypse', 'advanced', 'cybernetic', 'galactic'],
    
    'horror': ['horror', 'scary', 'terrifying', 'fear', 'nightmare', 'monster', 'ghost',
               'supernatural', 'haunt', 'demon', 'terrified', 'creature']
}

# Model training settings
DEFAULT_MODEL_SETTINGS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    },
    'bert': {
        'learning_rate': 2e-5,
        'weight_decay': 1e-2,
        'epochs': 5,
        'batch_size': 16,
        'early_stopping_patience': 3
    }
}

# Feature selection settings
FEATURE_SELECTION = {
    'n_features_to_select': 10,
    'cv': 5
}

# Preprocessing settings
TEXT_PREPROCESSING = {
    'remove_stopwords': True,
    'apply_stemming': True,
    'min_token_length': 2
}

# Evaluation settings
EVALUATION = {
    'test_size': 0.2,
    'random_state': 42,
    'metrics': ['accuracy', 'f1_micro', 'f1_macro', 'hamming_loss']
}

# Paths
PATHS = {
    'data_dir': './data',
    'output_dir': './results',
    'models_dir': './models',
    'visualizations_dir': './visualizations'
}
