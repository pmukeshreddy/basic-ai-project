"""
Example Movie Genre Classification Script
=======================================
This script demonstrates how to use the modularized movie genre classification code.
"""

import os
import pandas as pd
import numpy as np
import ast
from datetime import datetime

# Import our modules
import data_processing
import feature_extraction
import model_training
import evaluation
import config

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    """Main execution function"""
    # Set up directories
    data_dir = config.PATHS['data_dir']
    output_dir = config.PATHS['output_dir']
    models_dir = config.PATHS['models_dir']
    viz_dir = config.PATHS['visualizations_dir']
    
    # Ensure directories exist
    for directory in [data_dir, output_dir, models_dir, viz_dir]:
        ensure_dir(directory)
    
    # Set file paths
    data_path = os.path.join(data_dir, "movies_data.csv")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please download the dataset and place it in the data directory.")
        return
    
    print(f"Starting movie genre classification at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load and preprocess data
    print("\n--- Step 1: Loading and preprocessing data ---")
    df = data_processing.load_data(data_path)
    
    # If genres are stored as strings, convert to lists
    if 'genres' in df.columns and df['genres'].dtype == 'object':
        df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Print genre distribution
    genres_flat = [genre for sublist in df['genres'] for genre in sublist]
    genre_counts = pd.Series(genres_flat).value_counts()
    
    print("\nGenre distribution in the dataset:")
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count}")
    
    # Apply genre grouping
    df = data_processing.assign_genre_groups(df, 'genres', config.GENRE_TO_GROUP)
    
    # Clean data
    df_clean = data_processing.clean_data(df)
    
    # Step 2: Extract features
    print("\n--- Step 2: Extracting features ---")
    df_features = feature_extraction.extract_features(df_clean)
    
    # Step 3: Prepare data for modeling
    print("\n--- Step 3: Preparing data for modeling ---")
    X, y, label_names = data_processing.prepare_for_modeling(df_features, 'groups')
    X_train, X_test, y_train, y_test = data_processing.split_data(X, y)
    
    # Step 4: Train model
    print("\n--- Step 4: Training model ---")
    model, selected_features = model_training.train_random_forest(
        X_train, y_train, label_names, X.columns
    )
    
    # Save important features
    importance_df = model_training.get_feature_importance(model, X.columns)
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")
    
    # Step 5: Evaluate model
    print("\n--- Step 5: Evaluating model ---")
    metrics = evaluation.evaluate_model(model, X_test, y_test, label_names)
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'hamming_loss': metrics['hamming_loss'],
        'micro_f1': metrics['micro_f1'],
        'macro_f1': metrics['macro_f1'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model metrics saved to {metrics_path}")
    
    # Step 6: Create visualizations
    print("\n--- Step 6: Creating visualizations ---")
    visualizations = evaluation.create_visualizations(
        model, X, y, label_names, viz_dir
    )
    
    # Create feature importance by genre analysis
    heatmap_df, feature_viz = evaluation.analyze_feature_importance_by_genre(
        model, X, label_names, X.columns, viz_dir
    )
    
    # Save feature importance by genre
    heatmap_path = os.path.join(output_dir, "feature_importance_by_genre.csv")
    heatmap_df.to_csv(heatmap_path)
    print(f"Feature importance by genre saved to {heatmap_path}")
    
    # Print summary of visualizations
    print("\nVisualizations created:")
    for viz_name, viz_path in {**visualizations, **feature_viz}.items():
        print(f"  {viz_name}: {viz_path}")
    
    print(f"\nMovie genre classification completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
if __name__ == "__main__":
    main()
