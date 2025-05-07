"""
Data Processing Module
=====================
Functions for loading, cleaning, and preparing movie data for genre classification.
"""

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def load_data(file_path):
    """
    Load the movie dataset and perform initial conversions
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with genres converted to lists
    """
    df = pd.read_csv(file_path)
    
    # Convert genres from string to list if needed
    if 'genres' in df.columns and df['genres'].dtype == 'object':
        df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def clean_data(df):
    """
    Handle missing values, duplicates and inconsistent formatting
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    # Copy dataframe
    df_clean = df.copy()
    
    # Convert all list objects in object columns to tuples so they become hashable
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Now drop duplicates
    df_clean = df_clean.drop_duplicates()

    # Handle missing values
    df_clean['title'] = df_clean['title'].fillna('')
    if 'overview' in df_clean.columns:
        df_clean['overview'] = df_clean['overview'].fillna('')
    if 'release_date' in df_clean.columns:
        df_clean['release_date'] = df_clean['release_date'].fillna('2000-01-01')
    if 'vote_average' in df_clean.columns:
        df_clean['vote_average'] = df_clean['vote_average'].fillna(df_clean['vote_average'].mean())

    # Convert tuples back to lists where appropriate
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    print(f"Cleaned data: {df_clean.shape[0]} rows (removed {df.shape[0] - df_clean.shape[0]} duplicates)")
    return df_clean

def assign_genre_groups(df, genre_column='genres', group_mapping=None):
    """
    Map individual genres to broader genre groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    genre_column : str
        Column containing genre lists
    group_mapping : dict or None
        Mapping of genres to groups. If None, uses a default mapping.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with a new 'groups' column
    """
    df_with_groups = df.copy()
    
    # Default mapping if none provided
    if group_mapping is None:
        group_mapping = {
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
    
    # Function to assign groups based on the movie's genres
    def assign_groups(genre_list):
        groups = set()  # use a set to avoid duplicates
        for genre in genre_list:
            if genre in group_mapping:
                groups.add(group_mapping[genre])
        # Return a sorted list to maintain consistency
        return sorted(list(groups))
    
    # Create a new column 'groups' with the assigned group(s)
    df_with_groups['groups'] = df_with_groups[genre_column].apply(assign_groups)
    
    return df_with_groups

def prepare_for_modeling(df, target_column='groups'):
    """
    Transform data into format needed for machine learning models
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the column containing the target labels
    
    Returns:
    --------
    tuple
        (X, y, label_names) - features, target values, and label names
    """
    # Ensure target column contains lists
    df[target_column] = df[target_column].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Convert to multilabel format
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[target_column])
    label_names = mlb.classes_
    
    # Get features (all numeric columns except the target)
    X = df.select_dtypes(include=['number', 'bool']).drop(columns=[target_column], errors='ignore')
    
    # Convert boolean columns to int if present
    bool_cols = X.select_dtypes(include=['bool']).columns
    if not bool_cols.empty:
        X[bool_cols] = X[bool_cols].astype(int)
    
    print(f"Prepared data with {X.shape[1]} features and {len(label_names)} target classes")
    print(f"Label distribution: {dict(zip(label_names, y.sum(axis=0)))}")
    
    return X, y, label_names

def scale_features(df, numerical_columns=None):
    """
    Standardize numerical features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_columns : list or None
        List of numerical columns to scale. If None, uses all numeric columns.
    
    Returns:
    --------
    tuple
        (scaled_df, scaler) - dataframe with scaled features and the scaler object
    """
    df_scaled = df.copy()
    
    # If no columns specified, use all numeric columns
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter only columns that exist in the dataframe
    num_cols = [col for col in numerical_columns if col in df_scaled.columns]
    
    if num_cols:
        # Initialize scaler
        scaler = StandardScaler()
        
        # Scale numerical columns
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        print(f"Scaled {len(num_cols)} numerical features")
    else:
        print("No numerical columns found for scaling")
        scaler = None
    
    return df_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : np.array
        Target values
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # For multi-label, we need a stratification approach
    # Here we use a simple approach by taking the most common label for each row
    if y.shape[1] > 1:  # Multi-label case
        most_common_label = np.argmax(y, axis=1)
        stratify = most_common_label
    else:
        stratify = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print(f"Split data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(df, target_column='genres'):
    """
    Full preprocessing pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataframe
    target_column : str
        Name of the column containing target labels
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe ready for feature extraction
    """
    # 1. Clean data
    df_clean = clean_data(df)
    
    # 2. Assign genre groups if using original genres
    if target_column == 'genres':
        df_clean = assign_genre_groups(df_clean)
        target_column = 'groups'
    
    # 3. Scale features
    numerical_cols = [col for col in df_clean.columns 
                     if col.endswith('_sentiment') or 
                     col.endswith('_keywords') or 
                     col == 'vote_average']
    
    # Only scale if we have numerical columns
    if numerical_cols:
        df_scaled, _ = scale_features(df_clean, numerical_cols)
    else:
        df_scaled = df_clean
    
    print(f"Preprocessing completed. Data shape: {df_scaled.shape}")
    return df_scaled
