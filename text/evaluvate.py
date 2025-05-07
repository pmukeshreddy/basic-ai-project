"""
Evaluation Module
===============
Functions for evaluating model performance and creating visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, accuracy_score, hamming_loss, 
    f1_score, precision_score, recall_score
)
import os
import warnings

# Try importing plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    warnings.warn("Matplotlib and/or Seaborn not installed. Visualizations will not be available.")
    PLOTTING_AVAILABLE = False

def evaluate_model(model, X_test, y_test, label_names=None):
    """
    Evaluate model performance on test data
    
    Parameters:
    -----------
    model : trained model
        The model to evaluate
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray
        Test labels
    label_names : list or None
        Names of the target labels
    
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(X_test, pd.DataFrame):
        X_test_array = X_test.values
    else:
        X_test_array = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_array)
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    hl = hamming_loss(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Overall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Hamming Loss: {hl:.4f}")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    
    # Per-class metrics
    metrics = {
        'accuracy': accuracy,
        'hamming_loss': hl,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'per_class': {}
    }
    
    print("\nPer-class performance:")
    
    for i in range(y_test.shape[1]):
        label = str(i) if label_names is None else label_names[i]
        
        # Calculate metrics
        precision = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_test[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        
        # Store metrics
        metrics['per_class'][label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"  {label}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return metrics

def create_confusion_matrix(y_true, y_pred, label_names=None, output_dir=None):
    """
    Create confusion matrix for each label
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    label_names : list or None
        Names of the target labels
    output_dir : str or None
        Directory to save visualization
    
    Returns:
    --------
    dict
        Dictionary of file paths to saved visualizations
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping confusion matrix visualization.")
        return {}
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}
    
    for i in range(y_true.shape[1]):
        label = str(i) if label_names is None else label_names[i]
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {label}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure if output directory provided
        if output_dir is not None:
            filename = f"confusion_matrix_{label.replace(' ', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            visualizations[f'confusion_matrix_{label}'] = filepath
        
        plt.close()
    
    return visualizations

def plot_feature_importance(importance_df, output_dir=None, top_n=20):
    """
    Create feature importance visualization
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance scores
    output_dir : str or None
        Directory to save visualization
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    dict
        Dictionary of file paths to saved visualizations
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping feature importance visualization.")
        return {}
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    sns.barplot(data=top_features, x='Importance', y='Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir is not None:
        filepath = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(filepath)
        visualizations['feature_importance'] = filepath
    
    plt.close()
    
    return visualizations

def plot_label_distribution(y, label_names=None, output_dir=None):
    """
    Create label distribution visualization
    
    Parameters:
    -----------
    y : np.ndarray
        Labels
    label_names : list or None
        Names of the target labels
    output_dir : str or None
        Directory to save visualization
    
    Returns:
    --------
    dict
        Dictionary of file paths to saved visualizations
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping label distribution visualization.")
        return {}
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}
    
    # Calculate label distribution
    label_counts = y.sum(axis=0)
    
    # Create labels
    if label_names is None:
        label_names = [f"Label {i}" for i in range(y.shape[1])]
    
    # Create distribution plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_names, y=label_counts)
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure if output directory provided
    if output_dir is not None:
        filepath = os.path.join(output_dir, "label_distribution.png")
        plt.savefig(filepath)
        visualizations['label_distribution'] = filepath
    
    plt.close()
    
    return visualizations

def create_visualizations(model, X, y, label_names=None, output_dir=None):
    """
    Create and save all visualizations
    
    Parameters:
    -----------
    model : trained model
        The model to visualize
    X : np.ndarray or pd.DataFrame
        Features
    y : np.ndarray
        Labels
    label_names : list or None
        Names of the target labels
    output_dir : str or None
        Directory to save visualizations
    
    Returns:
    --------
    dict
        Dictionary of file paths to saved visualizations
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping all visualizations.")
        return {}
    
    visualizations = {}
    
    # Convert DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
    else:
        X_array = X
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # Make predictions
    y_pred = model.predict(X_array)
    
    # Create confusion matrix
    confusion_viz = create_confusion_matrix(y, y_pred, label_names, output_dir)
    visualizations.update(confusion_viz)
    
    # Create label distribution plot
    label_viz = plot_label_distribution(y, label_names, output_dir)
    visualizations.update(label_viz)
    
    # Create feature importance plot (if model supports it)
    if hasattr(model, 'estimators_'):
        # Get feature importances from all classifiers
        importances = np.zeros(X_array.shape[1])
        for estimator in model.estimators_:
            importances += estimator.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Create plot
        importance_viz = plot_feature_importance(importance_df, output_dir)
        visualizations.update(importance_viz)
    
    return visualizations

def analyze_feature_importance_by_genre(model, X, label_names, feature_names=None, output_dir=None):
    """
    Analyze feature importance breakdown by genre
    
    Parameters:
    -----------
    model : trained model
        The model to analyze
    X : np.ndarray or pd.DataFrame
        Features
    label_names : list
        Names of the target labels
    feature_names : list or None
        Names of the features
    output_dir : str or None
        Directory to save visualizations
    
    Returns:
    --------
    tuple
        (importance_matrix, visualizations) - importance scores and file paths
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping feature importance by genre.")
        return None, {}
    
    # Get feature names if X is a DataFrame
    if isinstance(X, pd.DataFrame) and feature_names is None:
        feature_names = X.columns.tolist()
    elif feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # Get top features
    importances = np.zeros(len(feature_names))
    for estimator in model.estimators_:
        importances += estimator.feature_importances_
    
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in top_indices]
    
    # Analyze feature importance for each genre
    genre_importances = {}
    importance_matrix = np.zeros((len(top_features), len(label_names)))
    
    for i, genre in enumerate(label_names):
        # Get feature importances for this genre
        importances = model.estimators_[i].feature_importances_
        genre_importances[genre] = pd.Series(importances, index=feature_names)
        
        # Store importances for top features
        for j, feat_idx in enumerate(top_indices):
            importance_matrix[j, i] = importances[feat_idx]
    
    # Create heatmap DataFrame
    heatmap_df = pd.DataFrame(
        importance_matrix, 
        index=top_features,
        columns=label_names
    )
    
    # Create visualization
    visualizations = {}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_df, annot=True, cmap='viridis')
        plt.title('Feature Importance by Genre')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, "feature_importance_by_genre.png")
        plt.savefig(filepath)
        visualizations['feature_importance_by_genre'] = filepath
        
        plt.close()
    
    return heatmap_df, visualizations
