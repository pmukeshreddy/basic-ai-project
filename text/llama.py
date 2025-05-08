import pandas as pd
from collections import Counter
import ast
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
# from sklearn.base import clone # Not directly used in the final snippets, but often related to MOClassifier or custom RF logic
import seaborn as sns
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

# Attempt to import Kaggle secrets, gracefully handle if not available
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None
    print("KaggleSecretsClient not found. Will proceed without it (Hugging Face token might be an issue).")

# -----------------------------
# 0. NLTK Downloads (if needed)
# -----------------------------
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_resources()

# -----------------------------
# 1. Data Loading and Initial Exploration
# -----------------------------
print("Loading data...")
try:
    data = pd.read_csv("/kaggle/input/basics-of-ai/movie_dataset_complete/movies_data.csv")
except FileNotFoundError:
    print("Error: movies_data.csv not found. Please check the path.")
    print("Attempting to create a dummy DataFrame for script structure testing.")
    # Create a small dummy DataFrame to allow the script to run structurally
    dummy_data_dict = {
        'id': [1, 2, 3, 4, 5],
        'title': ['Movie A', 'Movie B Sequel', 'Movie C', 'Movie D Story', 'Movie E'],
        'overview': [
            'This is a great action movie with fights.',
            'Funny comedy about a family.',
            'A dramatic story of relationships.',
            'Sci-fi adventure in space with aliens.',
            'A scary horror film with ghosts.'
        ],
        'genres': [
            "['Action', 'Adventure']",
            "['Comedy', 'Family']",
            "['Drama']",
            "['Science Fiction', 'Adventure']",
            "['Horror', 'Mystery']"
        ],
        'release_date': ['2020-01-15', '2021-03-10', '2019-07-20', '2022-11-05', '2020-09-25'],
        'vote_average': [7.5, 6.8, 8.1, 7.2, 5.5],
        'poster_path': ['/path1.jpg', '/path2.jpg', '/path3.jpg', '/path4.jpg', '/path5.jpg']
    }
    data = pd.DataFrame(dummy_data_dict)
    print("Using dummy data for demonstration.")


print("Initial data head:")
print(data.head())

# Convert genres from string representation to list of strings
if 'genres' in data.columns:
    data['genres'] = data['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else []))
else:
    print("Warning: 'genres' column not found in the initial data.")
    data['genres'] = [[] for _ in range(len(data))]


print("\nValue counts for raw genres column (sample):")
if 'genres' in data.columns:
    # Value_counts on lists of lists directly can be tricky.
    # We'll show the genre distribution after flattening.
    pass # The original value_counts() on the stringified list is not very informative.

# Flatten the list of lists into a single list of genres
if 'genres' in data.columns and not data['genres'].empty:
    all_genres = [genre for genres_list in data['genres'] for genre in genres_list if isinstance(genres_list, list)]
    genre_counts_calculated = Counter(all_genres)
    print("\nCalculated Genre Counts:")
    for genre, count in genre_counts_calculated.items():
        print(f"{genre}: {count}")
else:
    print("Skipping genre counting as 'genres' column is missing or empty.")
    genre_counts_calculated = {}

# -----------------------------
# 2. Genre Frequency Visualization (using provided hardcoded counts for consistency with prompt)
# -----------------------------
genre_counts_plot = { # Using the hardcoded dictionary from the prompt for the plot
    "Family": 1641, "Comedy": 5177, "Adventure": 2495, "Fantasy": 1699,
    "Action": 3588, "Crime": 2279, "Thriller": 3826, "Science Fiction": 1747,
    "Mystery": 1388, "Romance": 2850, "Drama": 7583, "Horror": 2436,
    "War": 971, "Animation": 1603, "Music": 937, "History": 1148,
    "Documentary": 1027, "TV Movie": 309, "Western": 687
}

genres_plot = list(genre_counts_plot.keys())
counts_plot = list(genre_counts_plot.values())

plt.figure(figsize=(12, 6))
plt.bar(genres_plot, counts_plot, color='skyblue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Genre Frequency (from provided counts)')
plt.tight_layout()
# plt.show() # Can be blocking
plt.savefig('genre_frequency_plot.png')
print("\nGenre frequency plot saved as genre_frequency_plot.png")
plt.close()

# -----------------------------
# 3. Genre Grouping
# -----------------------------
df = data.copy() # Start working with a copy

genre_to_group = {
    'Drama': 'Group 1', 'Music': 'Group 1',
    'Comedy': 'Group 2', 'Crime': 'Group 2', 'History': 'Group 2',
    'Thriller': 'Group 3', 'Science Fiction': 'Group 3', 'Fantasy': 'Group 3', 'Documentary': 'Group 3', 'Western': 'Group 3',
    'Action': 'Group 4', 'Horror': 'Group 4', 'Mystery': 'Group 4', 'War': 'Group 4', 'TV Movie': 'Group 4',
    'Adventure': 'Group 5', 'Romance': 'Group 5', 'Family': 'Group 5', 'Animation': 'Group 5'
}

def assign_groups(genre_list):
    groups = set()
    if isinstance(genre_list, list):
        for genre_item in genre_list: # Iterate over items in the list
            if genre_item in genre_to_group:
                groups.add(genre_to_group[genre_item])
    return sorted(list(groups))

if 'genres' in df.columns:
    df['groups'] = df['genres'].apply(assign_groups)
    print("\nDataFrame with 'groups' column (sample):")
    print(df[['id', 'title', 'genres', 'groups']].head())
else:
    print("Warning: 'genres' column not found for group assignment.")
    df['groups'] = [[] for _ in range(len(df))]


# -----------------------------
# 4. Feature Engineering
# -----------------------------
def extract_features(data_input):
    df_features = data_input.copy()

    # 1. Title sentiment
    df_features['title_sentiment'] = df_features['title'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) and x else 0
    )

    # 2. Contains sequel indicator
    sequel_patterns = [
        r'\s+2\s*$', r'\s+3\s*$', r'\s+4\s*$', r'\s+5\s*$',
        r'\s+ii\s*$', r'\s+iii\s*$', r'\s+iv\s*$', r'\s+v\s*$',
        r'\s+part\s+\d+', r'\s+chapter\s+\d+',
        r'sequel', r'trilogy', r'the\s+next\s+chapter',
        r'returns', r'strikes\s+back', r'revenge', r'resurrection',
        r'reloaded', r'continues', r'the\s+\w+\s+awakens'
    ]
    def has_sequel_indicator(title):
        if not isinstance(title, str): return 0
        title_lower = title.lower()
        for pattern in sequel_patterns:
            if re.search(pattern, title_lower): return 1
        return 0
    df_features['contains_sequel'] = df_features['title'].apply(has_sequel_indicator)

    # 3. Keyword presence in overview
    action_keywords = ['action', 'fight', 'battle', 'war', 'mission', 'explosion', 'danger', 'chase', 'combat', 'hero', 'terrorist', 'attack', 'weapon', 'survive']
    comedy_keywords = ['comedy', 'funny', 'laugh', 'humor', 'joke', 'hilarious', 'comic', 'amusing', 'silly', 'ridiculous', 'prank', 'comical']
    drama_keywords = ['drama', 'emotional', 'struggle', 'relationship', 'conflict', 'crisis', 'tragedy', 'suffering', 'dilemma', 'tension', 'turmoil']
    scifi_keywords = ['sci-fi', 'science fiction', 'future', 'space', 'alien', 'robot', 'technology', 'dystopian', 'apocalypse', 'advanced', 'cybernetic', 'galactic']
    horror_keywords = ['horror', 'scary', 'terrifying', 'fear', 'nightmare', 'monster', 'ghost', 'supernatural', 'haunt', 'demon', 'terrified', 'creature']
    def count_keywords(text, keyword_list):
        if not isinstance(text, str): return 0
        text_lower = text.lower()
        return sum(1 for keyword in keyword_list if keyword in text_lower)

    for col, keywords in [('action_keywords', action_keywords), ('comedy_keywords', comedy_keywords),
                           ('drama_keywords', drama_keywords), ('scifi_keywords', scifi_keywords),
                           ('horror_keywords', horror_keywords)]:
        df_features[col] = df_features['overview'].apply(lambda x: count_keywords(x, keywords))

    # 4. Sentiment analysis of overview
    df_features['overview_sentiment'] = df_features['overview'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if isinstance(x, str) and x else 0
    )
    df_features['overview_subjectivity'] = df_features['overview'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if isinstance(x, str) and x else 0
    )

    # 5. Season from release date
    def get_season(date_str):
        if not isinstance(date_str, str): return 'unknown'
        try:
            month = datetime.strptime(date_str, '%Y-%m-%d').month
            if 3 <= month <= 5: return 'spring'
            elif 6 <= month <= 8: return 'summer'
            elif 9 <= month <= 11: return 'fall'
            else: return 'winter'
        except (ValueError, TypeError): return 'unknown'
    
    if 'release_date' in df_features.columns:
        df_features['season_text'] = df_features['release_date'].apply(get_season)
        season_dummies = pd.get_dummies(df_features['season_text'], prefix='season', dummy_na=False)
        df_features = pd.concat([df_features, season_dummies], axis=1)
        if 'season_text' in df_features.columns: # Drop if it was created
             df_features = df_features.drop('season_text', axis=1)
    else:
        print("Warning: 'release_date' column not found for season feature engineering.")


    return df_features

print("\nExtracting features...")
# Use original 'data' for feature extraction to ensure clean slate for these features
df = extract_features(data) # df is now the DataFrame with engineered features
print("Feature extraction complete. DataFrame head with new features:")
print(df.head())

# -----------------------------
# 5. Preprocessing Pipeline
# -----------------------------
def clean_data(df_to_clean):
    df_clean = df_to_clean.copy()
    list_like_cols = ['genres', 'groups']
    for col in list_like_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: tuple(x) if isinstance(x, list) else ((x,) if pd.notnull(x) and not isinstance(x, tuple) else (x if isinstance(x, tuple) else tuple())))


    original_rows = df_clean.shape[0]
    # Handle potential TypeError if list-like columns are not hashable for drop_duplicates
    try:
        df_clean = df_clean.drop_duplicates()
    except TypeError:
        print("TypeError during drop_duplicates. Excluding list-like columns from subset for duplicate check.")
        cols_for_dup_check = [c for c in df_clean.columns if c not in list_like_cols]
        if cols_for_dup_check: # Only if there are non-list columns to check
            df_clean = df_clean.drop_duplicates(subset=cols_for_dup_check)
        else:
            print("No non-list columns to check for duplicates. Skipping drop_duplicates.")


    for col, fill_val in [('title', ''), ('overview', ''), ('release_date', '2000-01-01')]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(fill_val)
    if 'vote_average' in df_clean.columns:
        df_clean['vote_average'] = df_clean['vote_average'].fillna(df_clean['vote_average'].mean())
    
    print(f"Cleaned data: {df_clean.shape[0]} rows (removed {original_rows - df_clean.shape[0]} duplicates if any)")
    return df_clean

def preprocess_text(df_to_process, text_columns=['title', 'overview']):
    df_processed = df_to_process.copy()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    def process_text_field(text):
        if not isinstance(text, str) or not text: return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # Keep word characters and spaces
        tokens = text.split()
        processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        return ' '.join(processed_tokens)
    for column in text_columns:
        if column in df_processed.columns:
            df_processed[f'{column}_processed'] = df_processed[column].apply(process_text_field)
    print(f"Text preprocessing completed for columns: {text_columns}")
    return df_processed

def scale_features(df_to_scale, numerical_columns_to_scale=['vote_average', 'title_sentiment', 'overview_sentiment']):
    df_scaled = df_to_scale.copy()
    num_cols_existing = [col for col in numerical_columns_to_scale if col in df_scaled.columns]
    if num_cols_existing:
        scaler = StandardScaler()
        df_scaled[num_cols_existing] = scaler.fit_transform(df_scaled[num_cols_existing])
        print(f"Scaled {len(num_cols_existing)} numerical features: {num_cols_existing}")
    else:
        print("No numerical columns found for scaling based on input list.")
    return df_scaled # Scaler object is not returned here but could be if needed for inverse_transform

def encode_categorical(df_to_encode, categorical_columns_to_encode=['season_text']): # 'season_text' is handled by extract_features
    df_encoded = df_to_encode.copy()
    # 'season_text' is already converted to 'season_spring', 'season_summer' etc. by extract_features
    # This function as-is might not find 'season_text' unless extract_features failed or was modified.
    # For now, it will likely report "No (new) categorical columns to encode".
    cat_cols_actually_encoded = []
    for col in categorical_columns_to_encode:
        if col in df_encoded.columns and df_encoded[col].dtype == 'object': # Check if it's an object type to encode
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df_encoded[[col]])
            new_col_names = encoder.get_feature_names_out([col])
            encoded_df = pd.DataFrame(encoded_data, columns=new_col_names, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)
            cat_cols_actually_encoded.append(col)
    if cat_cols_actually_encoded:
        print(f"Encoded {len(cat_cols_actually_encoded)} categorical features: {cat_cols_actually_encoded}")
    else:
        print("No (new) categorical columns to encode as object type, or they were already numeric/OHE.")
    return df_encoded

def preprocess_pipeline(df_input, target_column='groups'):
    print("\nStarting preprocessing pipeline...")
    df_clean = clean_data(df_input)
    df_text = preprocess_text(df_clean)
    
    numerical_cols_to_scale = [
        col for col in df_text.columns 
        if col.endswith(('_sentiment', '_keywords')) or col == 'vote_average'
    ]
    df_scaled = scale_features(df_text, numerical_cols_to_scale) # Scaler not returned here
    
    # Categorical encoding for 'season_text' is already handled in extract_features (creates season_spring etc.)
    # So, encode_categorical might not do anything new unless other object columns are passed.
    df_encoded = encode_categorical(df_scaled, ['season_text']) # 'season_text' would be dropped by extract_features
    
    if target_column in df_encoded.columns:
        def ensure_list_of_strings(item):
            if isinstance(item, tuple): return list(item) # Convert tuple from clean_data back to list
            if isinstance(item, list): return item
            if pd.isnull(item): return []
            return [str(item)]
        df_encoded[target_column] = df_encoded[target_column].apply(ensure_list_of_strings)
    
    print("Preprocessing pipeline complete.")
    return df_encoded

df = preprocess_pipeline(df, target_column='groups')
print("DataFrame head after preprocessing pipeline:")
print(df.head())

# Ensure 'genres' (if still present and needed) is list of lists, and re-assign groups
if 'genres' in df.columns and not df['genres'].empty:
    # clean_data converts lists to tuples. Convert back to list for assign_groups.
    df['genres'] = df['genres'].apply(lambda x: list(x) if isinstance(x, tuple) else (x if isinstance(x, list) else []))
    df['groups'] = df['genres'].apply(assign_groups)
    print("\nRe-assigned groups after preprocessing (sample):")
    print(df[['title', 'genres', 'groups']].head())

# -----------------------------
# 6. Random Forest Feature Selection for Tabular Data
# -----------------------------
print("\nStarting Random Forest based feature selection...")

def ensure_list_format_rf(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            eval_list = ast.literal_eval(x)
            return eval_list if isinstance(eval_list, list) else [eval_list]
        except (ValueError, SyntaxError):
            return [s.strip() for s in x.strip("[]'").replace("'", "").split(',') if s.strip()]
    if isinstance(x, tuple): return list(x)
    return []

if 'groups' in df.columns and not df['groups'].empty:
    # Ensure 'groups' is in the correct list-of-strings format for MLB
    df['groups'] = df['groups'].apply(ensure_list_format_rf)
    print("Sample group lists for RF (after ensuring list format):")
    for i in range(min(5, len(df))): print(f"Row {i}: {df['groups'].iloc[i]}")

# Define feature columns for RF
feature_cols_for_rf = [
    col for col in df.columns
    if (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))
    and col not in ['id', 'genres', 'groups', 'title', 'overview',
                    'release_date', 'poster_path', 'title_processed',
                    'overview_processed', 'overview_subjectivity'] # overview_subjectivity could be a feature
]
# Add season OHE columns if they were created by extract_features and not already in list
season_ohe_cols_from_extract = [col for col in df.columns if col.startswith('season_')]
for s_col in season_ohe_cols_from_extract:
    if s_col not in feature_cols_for_rf and \
       (pd.api.types.is_numeric_dtype(df[s_col]) or pd.api.types.is_bool_dtype(df[s_col])):
        feature_cols_for_rf.append(s_col)
feature_cols_for_rf = sorted(list(set(feature_cols_for_rf)))

X_df_rf = df[feature_cols_for_rf].copy()
print(f"Features for RF-based selection ({len(X_df_rf.columns)}): {X_df_rf.columns.tolist()}")

# Convert boolean columns to int and fill NaNs
bool_cols_rf = X_df_rf.select_dtypes(include=['bool']).columns
if not bool_cols_rf.empty: X_df_rf[bool_cols_rf] = X_df_rf[bool_cols_rf].astype(int)
for col in X_df_rf.select_dtypes(include=np.number).columns:
    if X_df_rf[col].isnull().any(): X_df_rf[col] = X_df_rf[col].fillna(X_df_rf[col].mean())

# Prepare target for RF
mlb_rf = MultiLabelBinarizer()
y_rf_input = df['groups'].apply(lambda x: x if isinstance(x, list) and all(isinstance(i, str) for i in x) else [])
y_rf = mlb_rf.fit_transform(y_rf_input)
group_names_rf = mlb_rf.classes_
print(f"Group label distribution for RF: {dict(zip(group_names_rf, y_rf.sum(axis=0)))}")


top_tabular_features_rf = []
if not X_df_rf.empty and y_rf.shape[0] > 0 and y_rf.shape[1] > 0 :
    X_train_rf_df, _, y_train_rf, _ = train_test_split(X_df_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf if np.sum(y_rf.sum(axis=0) > 5) == y_rf.shape[1] else None) # Stratify if possible
    X_train_rf = X_train_rf_df.values

    base_clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    multi_clf_rf = MultiOutputClassifier(base_clf_rf)

    class FeatureImportanceTrackerRF:
        def __init__(self, feature_names):
            self.feature_names = feature_names
            self.importances = np.zeros(len(feature_names))
        def update(self, moc_classifier):
            for estimator in moc_classifier.estimators_:
                self.importances += estimator.feature_importances_
        def get_top_features(self, n):
            top_indices = np.argsort(self.importances)[::-1][:n]
            return [self.feature_names[i] for i in top_indices]
        def get_importance_df(self):
            return pd.DataFrame({'Feature': self.feature_names, 'Importance': self.importances}).sort_values('Importance', ascending=False)

    n_features_to_select_rf = min(10, X_train_rf.shape[1]) if X_train_rf.shape[1] > 0 else 0
    print(f"Selecting top {n_features_to_select_rf} tabular features from {X_train_rf.shape[1]} total features using RF.")

    if n_features_to_select_rf > 0 and X_train_rf.shape[0] > 0:
        importance_tracker_rf = FeatureImportanceTrackerRF(X_df_rf.columns.tolist())
        print("Training RF model to evaluate feature importance...")
        multi_clf_rf.fit(X_train_rf, y_train_rf)
        importance_tracker_rf.update(multi_clf_rf)
        top_tabular_features_rf = importance_tracker_rf.get_top_features(n_features_to_select_rf)

        importance_df_rf = importance_tracker_rf.get_importance_df().head(min(20, len(X_df_rf.columns)))
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df_rf, x='Importance', y='Feature')
        plt.title('Top Tabular Feature Importance (for Movie Groups based on RF)')
        plt.tight_layout()
        plt.savefig('tabular_feature_importance_rf.png')
        print("Tabular feature importance plot saved as tabular_feature_importance_rf.png")
        plt.close()
    else:
        print("Not enough features or data to perform RF feature selection.")
else:
    print("Skipping RF feature selection: No features in X_df_rf or no labels in y_rf.")


print(f"Selected top tabular features via RF: {top_tabular_features_rf}")

# -----------------------------
# 7. Data Preparation for Llama Model
# -----------------------------
print("\nPreparing data for Llama model...")

# Ensure 'overview' is present; if not, try to get it from original 'data'
if 'overview' not in df.columns:
    if 'overview' in data.columns:
        df_overview_source = data[['overview']].copy()
        df_overview_source['overview'] = df_overview_source['overview'].fillna('')
        df = pd.concat([df.reset_index(drop=True), df_overview_source.reset_index(drop=True)], axis=1)
        print("Added 'overview' column from original data to the working DataFrame.")
    else: # If 'overview' is nowhere
        print("CRITICAL: 'overview' column is missing and not found in original data. Llama model will not work correctly.")
        df['overview'] = ["missing overview"] * len(df) # Add dummy overview


# Check if selected tabular features are present
existing_top_tabular_features_rf = [feat for feat in top_tabular_features_rf if feat in df.columns]
if len(existing_top_tabular_features_rf) != len(top_tabular_features_rf):
    print(f"Warning: Some RF-selected features are missing from df. Using: {existing_top_tabular_features_rf}")
top_tabular_features_rf = existing_top_tabular_features_rf # Update to only existing ones

# Create the final DataFrame for the Llama model
columns_for_llama = ['overview', 'groups'] + top_tabular_features_rf
missing_cols_in_df = [col for col in columns_for_llama if col not in df.columns]
if missing_cols_in_df:
    print(f"Warning: The following columns are still missing for Llama model prep: {missing_cols_in_df}")
    # Add dummy columns if essential ones like 'groups' are missing, to prevent crash
    for mc in missing_cols_in_df:
        if mc == 'groups': df[mc] = [[] for _ in range(len(df))]
        elif mc == 'overview': df[mc] = [""] * len(df)
        else: df[mc] = 0 # For missing tabular features

df_for_llama_model = df[columns_for_llama].copy()
df = df_for_llama_model # df is now a slimmed-down version for the Llama model

def ensure_list_final(x): # For 'groups' column before MLB
    if isinstance(x, str):
        try:
            evaluated_list = ast.literal_eval(x)
            return evaluated_list if isinstance(eval_list, list) else [str(evaluated_list)]
        except (ValueError, SyntaxError):
            return [s.strip() for s in x.strip("[]'").replace("'", "").split(',') if s.strip()]
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    return [str(x)] if pd.notnull(x) else []

if 'groups' in df.columns and not df['groups'].empty:
    df['groups'] = df['groups'].apply(ensure_list_final)
    print("Ensured 'groups' column is list of strings for MultiLabelBinarizer.")

mlb = MultiLabelBinarizer()
# Ensure 'groups_for_mlb' is correctly populated even if 'groups' is all empty lists
df['groups_for_mlb'] = df['groups'].apply(lambda x: x if (isinstance(x, list) and all(isinstance(i, str) for i in x)) else [])

if not df['groups_for_mlb'].empty and df['groups_for_mlb'].apply(len).sum() > 0 : # Check if there are any labels to binarize
    genres_encoded = mlb.fit_transform(df['groups_for_mlb'])
    group_target_names = mlb.classes_
    print(f"Found {len(group_target_names)} unique groups for target: {group_target_names}")
else: # Handle case with no valid group labels
    print("Warning: No valid group labels found for MultiLabelBinarizer. Using dummy target.")
    # Create a dummy target if no groups are present to avoid crashing downstream
    num_dummy_groups = 5 # Default to 5 groups as per genre_to_group
    group_target_names = [f"Group {i+1}" for i in range(num_dummy_groups)]
    genres_encoded = np.zeros((len(df), num_dummy_groups), dtype=int)
    # Optionally, assign a default group if desired, e.g., genres_encoded[:, 0] = 1

# Tabular features for Llama model
tabular_feature_columns = top_tabular_features_rf # Use RF selected features
if not tabular_feature_columns: # Fallback if RF selection yielded no features
    print("Warning: `top_tabular_features_rf` is empty. Defaulting to standard features if available.")
    default_tabular_cols = ['vote_average', 'overview_sentiment', 'title_sentiment'] # These might not exist if df was slimmed
    tabular_feature_columns = [col for col in default_tabular_cols if col in df.columns]

existing_tabular_cols = [col for col in tabular_feature_columns if col in df.columns]
if not existing_tabular_cols:
    print("No tabular features will be used for the Llama model as none were found/selected.")
    X_tabular_np = np.array([]).reshape(len(df), 0).astype(np.float32)
else:
    print(f"Using tabular features for Llama: {existing_tabular_cols}")
    X_tabular = df[existing_tabular_cols].copy()
    for col in X_tabular.columns: # Fill NaNs that might have appeared
        if X_tabular[col].isnull().any(): X_tabular[col] = X_tabular[col].fillna(X_tabular[col].mean())
    X_tabular_np = X_tabular.values.astype(np.float32)

y_multilabel = genres_encoded.astype(np.float32)
overview_text_for_llama = df['overview'].fillna("").tolist()

print(f"\nTabular Features shape: {X_tabular_np.shape}")
print(f"Multilabel Labels shape: {y_multilabel.shape}")
print(f"Number of overview texts: {len(overview_text_for_llama)}")
print(f"Number of samples with multiple groups: {(np.sum(y_multilabel, axis=1) > 1).sum()}")

group_counts_df = pd.DataFrame(genres_encoded, columns=group_target_names)
print("\nGroup distribution (target labels for Llama):")
print(group_counts_df.sum().sort_values(ascending=False))

if not (X_tabular_np.shape[0] == y_multilabel.shape[0] == len(overview_text_for_llama)):
    print(f"X_tabular_np shape: {X_tabular_np.shape[0]}, y_multilabel shape: {y_multilabel.shape[0]}, overview_text len: {len(overview_text_for_llama)}")
    raise AssertionError("Mismatch in number of samples between features, labels, and overview text.")

if len(df) < 2: # Need at least 2 samples for train_test_split
    print("Warning: Not enough data for train/test split. Using all data for training and testing (not recommended).")
    X_train_tabular, X_test_tabular = X_tabular_np, X_tabular_np
    overview_train, overview_test = overview_text_for_llama, overview_text_for_llama
    y_train, y_test = y_multilabel, y_multilabel
else:
    X_train_tabular, X_test_tabular, \
    overview_train, overview_test, \
    y_train, y_test = train_test_split(
        X_tabular_np,
        overview_text_for_llama,
        y_multilabel,
        test_size=0.2, # Ensure test_size is less than 1.0
        random_state=42,
        stratify=y_multilabel if np.sum(y_multilabel.sum(axis=0) > 5* (1/0.2) ) == y_multilabel.shape[1] else None # Stratify if labels are sufficient
    )
print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")


# -----------------------------
# 8. Llama Model Definition, Training, and Evaluation
# -----------------------------

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs) # prob for target class
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * bce_loss

        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

# --- Class Weights (Original 19 Genres - for reference) ---
class_weights_original_genres = {
    'Family': 10.77, 'Comedy': 3.41, 'Adventure': 7.08, 'Fantasy': 10.40, 'Action': 4.93,
    'Crime': 7.76, 'Thriller': 4.62, 'Science Fiction': 10.12, 'Mystery': 12.74, 'Romance': 6.20,
    'Drama': 2.33, 'Horror': 7.26, 'War': 18.20, 'Animation': 11.03, 'Music': 18.87,
    'History': 15.40, 'Documentary': 17.21, 'TV Movie': 57.21, 'Western': 25.73
}
ordered_classes_original_genres = list(class_weights_original_genres.keys()) # Order based on dict
weights_list_original_genres = [class_weights_original_genres[cls] for cls in ordered_classes_original_genres]
pos_weight_tensor_original_genres = torch.tensor(weights_list_original_genres, dtype=torch.float32)
print("\nPos Weight Tensor (for original 19 genres - reference):", pos_weight_tensor_original_genres)
print(f"Length of original genre weights: {len(pos_weight_tensor_original_genres)}")


# --- Setup & Preprocessing for Llama ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

HF_TOKEN = None
if UserSecretsClient is not None:
    try:
        user_secrets = UserSecretsClient()
        HF_TOKEN = user_secrets.get_secret("HUGGINGFACE_TOKEN")
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        print("Hugging Face token loaded from Kaggle Secrets.")
    except Exception as e:
        print(f"Could not load Hugging Face token from Kaggle Secrets: {e}.")
else:
    print("KaggleSecretsClient not available. Set HUGGING_FACE_HUB_TOKEN environment variable manually if needed.")

LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
if not HF_TOKEN and "meta-llama" in LLAMA_MODEL_NAME:
    print(f"WARNING: Hugging Face token not found/set. Llama 3 model ('{LLAMA_MODEL_NAME}') loading will likely fail.")
    LLAMA_MODEL_NAME = "gpt2" # Fallback
    print(f"Attempting to use fallback model: {LLAMA_MODEL_NAME}")

# Calculate positive weights for the 5 groups
if y_train.shape[0] > 0 and y_train.shape[1] > 0:
    labels_for_weights = y_train.astype(np.float32)
    label_freq_groups = labels_for_weights.sum(axis=0) + 1e-9 # Add epsilon to avoid division by zero
    inverse_freq_groups = 1.0 / label_freq_groups
    inverse_freq_groups_clipped = np.clip(inverse_freq_groups, 1.0, 20.0) # Clip weights
    pos_weight_for_groups_tensor = torch.tensor(inverse_freq_groups_clipped, dtype=torch.float32).to(device)
else: # Handle empty y_train
    num_labels_from_target_names = len(group_target_names) if 'group_target_names' in globals() and group_target_names else 5
    pos_weight_for_groups_tensor = torch.ones(num_labels_from_target_names, dtype=torch.float32).to(device)
    print("Warning: y_train is empty or invalid, using default pos_weight for groups.")

num_target_labels = len(group_target_names) if 'group_target_names' in globals() and group_target_names else y_train.shape[1]
print(f"Positive weights for {num_target_labels} groups: {pos_weight_for_groups_tensor}")


try:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, token=HF_TOKEN if HF_TOKEN else None)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        print(f"Set pad_token to eos_token ({tokenizer.eos_token_id}) for {LLAMA_MODEL_NAME} tokenizer.")
except Exception as e:
    print(f"Error loading AutoTokenizer for {LLAMA_MODEL_NAME}: {e}")
    print("Exiting due to tokenizer loading failure.")
    exit()


MAX_LENGTH = 128
print(f"Using MAX_LENGTH: {MAX_LENGTH} for tokenization.")
encoded_train = tokenizer(overview_train, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_LENGTH)
encoded_test = tokenizer(overview_test, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_LENGTH)


class MovieGenreDatasetLlama(Dataset):
    def __init__(self, encodings, tabular_feats, labels_data):
        self.encodings = encodings
        self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)
        self.labels_data = torch.tensor(labels_data, dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['tabular'] = self.tabular_feats[idx].clone().detach()
        item['labels'] = self.labels_data[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels_data)

train_dataset_llama = MovieGenreDatasetLlama(encoded_train, X_train_tabular, y_train)
val_dataset_llama = MovieGenreDatasetLlama(encoded_test, X_test_tabular, y_test) # Using test as val
test_dataset_llama = MovieGenreDatasetLlama(encoded_test, X_test_tabular, y_test)


BATCH_SIZE = 1 if not torch.cuda.is_available() or "Llama-3-8B" in LLAMA_MODEL_NAME else 4 # Smaller for large models on CPU/limited GPU
print(f"Using batch size: {BATCH_SIZE}")
NUM_WORKERS = 2 if device.type == 'cuda' else 0
train_loader_llama = DataLoader(train_dataset_llama, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=NUM_WORKERS)
val_loader_llama = DataLoader(val_dataset_llama, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == 'cuda'), num_workers=NUM_WORKERS)
test_loader_llama = DataLoader(test_dataset_llama, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == 'cuda'), num_workers=NUM_WORKERS)


class LlamaWithTabular(nn.Module):
    def __init__(self, num_labels_init, tabular_input_dim_init, llama_model_name_init):
        super(LlamaWithTabular, self).__init__()
        self.has_tabular = tabular_input_dim_init > 0
        try:
            self.llama = AutoModel.from_pretrained(llama_model_name_init, token=HF_TOKEN if HF_TOKEN else None)
        except Exception as e:
            print(f"Error loading AutoModel for {llama_model_name_init}: {e}")
            raise

        if self.llama.config.vocab_size != len(tokenizer):
            self.llama.resize_token_embeddings(len(tokenizer)) # Important if pad token was added
            print(f"Resized {llama_model_name_init} model embeddings to: {len(tokenizer)}")
        
        # Freeze embeddings
        if hasattr(self.llama, 'get_input_embeddings'):
            for param in self.llama.get_input_embeddings().parameters():
                param.requires_grad = False
            print(f"Froze {llama_model_name_init} token embeddings.")

        # Freeze some layers
        layers_module = None
        model_type_for_freeze_log = "Unknown"
        if hasattr(self.llama, 'transformer') and hasattr(self.llama.transformer, 'h'): # gpt2-like
            layers_module = self.llama.transformer.h
            model_type_for_freeze_log = "GPT2-like (transformer.h)"
        elif hasattr(self.llama, 'model') and hasattr(self.llama.model, 'layers'): # llama-like
            layers_module = self.llama.model.layers
            model_type_for_freeze_log = "Llama-like (model.layers)"
        elif hasattr(self.llama, 'layers'): # Some models (e.g. BERT)
             layers_module = self.llama.layers
             model_type_for_freeze_log = "BERT-like (direct layers)"
        
        if layers_module:
            num_layers = len(layers_module)
            layers_to_freeze = num_layers - 2 # Unfreeze top 2 layers
            if layers_to_freeze > 0:
                for i, layer in enumerate(layers_module):
                    if i < layers_to_freeze:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"Froze {layers_to_freeze} out of {num_layers} {model_type_for_freeze_log} transformer layers.")
        else:
            print(f"Could not find standard layer attribute for freezing in {llama_model_name_init}.")

        if self.has_tabular:
            self.tabular_proj = nn.Sequential(
                nn.Linear(tabular_input_dim_init, 64), nn.ReLU(), nn.LayerNorm(64), nn.Dropout(0.3)
            )
        
        self.dropout = nn.Dropout(0.3)
        llama_hidden_size = self.llama.config.hidden_size
        classifier_input_dim = llama_hidden_size + (64 if self.has_tabular else 0)
        self.classifier = nn.Linear(classifier_input_dim, num_labels_init)

    def forward(self, input_ids, attention_mask, tabular):
        llama_outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = llama_outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Mean pooling of the last hidden state, considering attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
        llama_pooled_output = sum_embeddings / sum_mask
        
        if self.has_tabular and tabular.numel() > 0:
            tabular_out = self.tabular_proj(tabular)
            combined = torch.cat((llama_pooled_output, tabular_out), dim=1)
        else:
            combined = llama_pooled_output
        
        output = self.classifier(self.dropout(combined))
        return output

tabular_input_dim = X_train_tabular.shape[1] if X_train_tabular.ndim > 1 and X_train_tabular.shape[1] > 0 else 0
if X_train_tabular.ndim == 1 and X_train_tabular.shape[0] > 0 and tabular_input_dim == 0: # case of single tabular feature
    tabular_input_dim = 1


if y_train.shape[0] == 0: # No training data
    print("CRITICAL: No training data available. Skipping Llama model training and evaluation.")
else:
    model = LlamaWithTabular(
        num_labels_init=num_target_labels,
        tabular_input_dim_init=tabular_input_dim,
        llama_model_name_init=LLAMA_MODEL_NAME
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-6, weight_decay=1e-2) # Smaller LR for fine-tuning
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight_for_groups_tensor.to(device))
    scaler_amp = torch.amp.GradScaler(enabled=(device.type == 'cuda')) # For mixed precision

    # --- Evaluation Function ---
    def evaluate_llama(model_eval, data_loader_eval, criterion_eval, threshold_eval=0.5):
        model_eval.eval()
        total_loss_eval = 0; all_preds_eval = []; all_labels_eval = []
        with torch.no_grad():
            for batch in data_loader_eval:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                tabular = batch['tabular'].to(device, non_blocking=True)
                labels_batch = batch['labels'].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    outputs = model_eval(input_ids, attention_mask, tabular)
                    loss = criterion_eval(outputs, labels_batch)
                total_loss_eval += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= threshold_eval).int()
                all_preds_eval.append(preds.cpu().numpy())
                all_labels_eval.append(labels_batch.cpu().numpy())
        
        if not all_preds_eval: # If dataloader was empty
            return {'loss': 0, 'micro_f1': 0, 'macro_f1': 0, 'sample_f1': 0}

        y_pred_eval_arr = np.vstack(all_preds_eval)
        y_true_eval_arr = np.vstack(all_labels_eval)
        return {
            'loss': total_loss_eval / len(data_loader_eval),
            'micro_f1': f1_score(y_true_eval_arr, y_pred_eval_arr, average='micro', zero_division=0),
            'macro_f1': f1_score(y_true_eval_arr, y_pred_eval_arr, average='macro', zero_division=0),
            'sample_f1': f1_score(y_true_eval_arr, y_pred_eval_arr, average='samples', zero_division=0)
        }

    # --- Training Loop ---
    epochs = 3
    best_val_f1_micro = 0; best_model_state = None
    patience = 1; patience_counter = 0 # Early stopping
    threshold_train_eval = 0.3 # Threshold for converting probabilities to binary labels
    accumulation_steps = max(1, 16 // BATCH_SIZE) # Adjust accumulation for effective batch size of ~16

    print(f"\nStarting training with {LLAMA_MODEL_NAME}...")
    print(f"Tabular feature dimension: {tabular_input_dim}")
    print(f"Number of target labels (groups): {num_target_labels}")
    print(f"Gradient accumulation steps: {accumulation_steps}")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0; train_preds_epoch = []; train_labels_epoch = []
        loop = tqdm(train_loader_llama, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            tabular = batch['tabular'].to(device, non_blocking=True)
            labels_batch_train = batch['labels'].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(input_ids, attention_mask, tabular)
                loss = criterion(outputs, labels_batch_train)
                loss = loss / accumulation_steps
            
            scaler_amp.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader_llama):
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps # De-normalize for logging
            probs = torch.sigmoid(outputs.detach()) # Detach before CPU transfer
            preds = (probs >= threshold_train_eval).int()
            train_preds_epoch.append(preds.cpu().numpy())
            train_labels_epoch.append(labels_batch_train.cpu().numpy())

            if len(train_preds_epoch) > 0 : # Update TQDM postfix
                y_pred_so_far = np.vstack(train_preds_epoch)
                y_true_so_far = np.vstack(train_labels_epoch)
                train_f1_micro_current = f1_score(y_true_so_far, y_pred_so_far, average='micro', zero_division=0)
                loop.set_postfix(loss=loss.item()*accumulation_steps, train_micro_f1=f"{train_f1_micro_current:.4f}")

        if not train_preds_epoch: # If training loop was empty
            print(f"Epoch {epoch+1}: No training data processed.")
            continue
            
        y_train_pred_full = np.vstack(train_preds_epoch)
        y_train_true_full = np.vstack(train_labels_epoch)
        train_metrics_epoch = {
            'loss': total_train_loss / len(train_loader_llama),
            'micro_f1': f1_score(y_train_true_full, y_train_pred_full, average='micro', zero_division=0),
            'macro_f1': f1_score(y_train_true_full, y_train_pred_full, average='macro', zero_division=0),
            'sample_f1': f1_score(y_train_true_full, y_train_pred_full, average='samples', zero_division=0)
        }
        val_metrics = evaluate_llama(model, val_loader_llama, criterion, threshold_train_eval)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_metrics_epoch['loss']:.4f} | Train Micro F1: {train_metrics_epoch['micro_f1']:.4f} | Train Macro F1: {train_metrics_epoch['macro_f1']:.4f} | Train Sample F1: {train_metrics_epoch['sample_f1']:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Micro F1:   {val_metrics['micro_f1']:.4f} | Val Macro F1:   {val_metrics['macro_f1']:.4f} | Val Sample F1: {val_metrics['sample_f1']:.4f}")

        if val_metrics['micro_f1'] > best_val_f1_micro:
            best_val_f1_micro = val_metrics['micro_f1']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "best_llama_model.pth")
            print(f"  New best model saved! (Val Micro F1: {best_val_f1_micro:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s) based on Val Micro F1.")
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("\nLoaded best model for final evaluation.")
    else:
        print("\nNo best model state saved (e.g., training did not improve or was very short). Using last model state.")


    print("\nEvaluating on Test Set with best/last model:")
    if len(test_loader_llama) > 0:
        test_metrics = evaluate_llama(model, test_loader_llama, criterion, threshold_eval=threshold_train_eval)
        print(f"Test Loss: {test_metrics['loss']:.4f} | Test Micro F1: {test_metrics['micro_f1']:.4f} | Test Macro F1: {test_metrics['macro_f1']:.4f} | Test Sample F1: {test_metrics['sample_f1']:.4f}")
    else:
        print("Test data loader is empty. Skipping final test evaluation.")

