"""
Model Training Module
====================
Functions for training different types of models for genre classification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import warnings

# Try importing deep learning libraries but don't fail if they're not available
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    from torch.optim import AdamW
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not installed. BERT model will not be available.")

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    warnings.warn("Transformers library not installed. BERT model will not be available.")

def train_random_forest(X_train, y_train, label_names=None, feature_names=None):
    """
    Train a Random Forest classifier with feature selection
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    label_names : list or None
        Names of the target labels
    feature_names : list or None
        Names of the features
    
    Returns:
    --------
    tuple
        (model, selected_features) - trained model and list of selected features
    """
    print("Training Random Forest classifier with feature selection...")
    
    # Get feature names if X_train is a DataFrame
    if isinstance(X_train, pd.DataFrame) and feature_names is None:
        feature_names = X_train.columns.tolist()
    
    # Convert DataFrame to numpy array if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    # Base classifier
    base_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Create a MultiOutputClassifier for multilabel classification
    multi_clf = MultiOutputClassifier(base_clf)
    
    # Apply class weights through sample weights
    if label_names is not None:
        # Calculate inverse frequency weights
        n_samples = len(X_train)
        label_counts = y_train.sum(axis=0)
        weights = {}
        
        for i, label in enumerate(label_names):
            label_count = label_counts[i]
            if label_count > 0:
                weights[label] = n_samples / label_count
            else:
                weights[label] = 0
        
        # Apply weights to samples
        sample_weights = np.ones(len(y_train))
        for i, label in enumerate(label_names):
            # Get samples with this label
            has_label = y_train[:, i] == 1
            # Add weight for this label
            sample_weights[has_label] *= weights[label]
        
        # Normalize weights
        sample_weights = sample_weights / np.mean(sample_weights)
        
        # Train with weights
        multi_clf.fit(X_train_array, y_train, sample_weight=sample_weights)
    else:
        # Train without weights
        multi_clf.fit(X_train_array, y_train)
    
    # Feature importance tracking
    importances = np.zeros(X_train_array.shape[1])
    for estimator in multi_clf.estimators_:
        importances += estimator.feature_importances_
    
    # Get top features
    n_features_to_select = min(10, X_train_array.shape[1])
    top_indices = np.argsort(importances)[::-1][:n_features_to_select]
    
    if feature_names is not None:
        selected_features = [feature_names[i] for i in top_indices]
    else:
        selected_features = top_indices.tolist()
    
    print(f"Selected top {n_features_to_select} features: {selected_features}")
    
    return multi_clf, selected_features

def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model
    
    Parameters:
    -----------
    model : MultiOutputClassifier
        Trained model
    feature_names : list
        Names of the features
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance scores
    """
    # Get feature importances from all classifiers
    importances = np.zeros(len(feature_names))
    for estimator in model.estimators_:
        importances += estimator.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def train_multioutput_classifier(X_train, y_train):
    """
    Train a basic multi-output classifier
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    
    Returns:
    --------
    MultiOutputClassifier
        Trained model
    """
    print("Training MultiOutput classifier...")
    
    # Convert DataFrame to numpy array if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    # Base classifier
    base_clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Create a MultiOutputClassifier for multilabel classification
    multi_clf = MultiOutputClassifier(base_clf)
    
    # Train the model
    multi_clf.fit(X_train_array, y_train)
    
    return multi_clf

def train_bert_model(X_train, y_train, text_data, label_names=None, epochs=5):
    """
    Train a BERT model for text classification
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray
        Training labels
    text_data : list
        List of text data (e.g., movie overviews)
    label_names : list or None
        Names of the target labels
    epochs : int
        Number of training epochs
    
    Returns:
    --------
    object
        Trained model
    """
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        print("BERT model training requires PyTorch and Transformers libraries.")
        print("Please install with: pip install torch transformers")
        return None
    
    print("Training BERT-based model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert text to strings
    text_data = [str(text) if text is not None else "" for text in text_data]
    
    # Tokenize text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
    
    # Convert tabular features to tensor
    if isinstance(X_train, pd.DataFrame):
        tabular_features = X_train.values
    else:
        tabular_features = X_train
    
    tabular_features = tabular_features.astype(np.float32)
    
    # Class weights
    if label_names is not None:
        label_freq = y_train.sum(axis=0)
        inverse_freq = 1.0 / (label_freq + 1e-6)
        inverse_freq_clipped = np.clip(inverse_freq, 1.0, 10.0)
        inverse_freq_tensor = torch.tensor(inverse_freq_clipped, dtype=torch.float32).to(device)
    else:
        inverse_freq_tensor = None
    
    # Create dataset
    class MovieDataset(Dataset):
        def __init__(self, encodings, tabular_feats, labels):
            self.encodings = encodings
            self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['tabular'] = self.tabular_feats[idx]
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)
    
    # Create dataset
    dataset = MovieDataset(encoded, tabular_features, y_train)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
    
    # Define model
    class BERTWithTabular(nn.Module):
        def __init__(self, num_labels, tabular_input_dim):
            super(BERTWithTabular, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
            # Freeze all encoder layers except the last 2
            for layer in self.bert.encoder.layer[:10]:
                for param in layer.parameters():
                    param.requires_grad = False
                    
            self.tabular_proj = nn.Sequential(
                nn.Linear(tabular_input_dim, 32),
                nn.ReLU(),
                nn.LayerNorm(32),
                nn.Dropout(0.5)
            )
            self.dropout = nn.Dropout(0.5)
            self.classifier = nn.Linear(768 + 32, num_labels)
            
        def forward(self, input_ids, attention_mask, tabular):
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            tabular_out = self.tabular_proj(tabular)
            combined = torch.cat((bert_out, tabular_out), dim=1)
            output = self.classifier(self.dropout(combined))
            return output
    
    # Initialize model
    model = BERTWithTabular(num_labels=y_train.shape[1], tabular_input_dim=tabular_features.shape[1])
    model.to(device)
    
    # Define loss function
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.reduction = reduction
            self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

        def forward(self, inputs, targets):
            bce_loss = self.bce(inputs, targets)
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets == 1, probs, 1 - probs)
            focal_factor = (1 - pt) ** self.gamma
            loss = focal_factor * bce_loss

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
    
    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    criterion = FocalLoss(gamma=2.0, pos_weight=inverse_freq_tensor)
    
    # Training loop
    best_val_f1 = 0
    best_model_state = None
    patience = 3
    patience_counter = 0
    threshold = 0.3
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tabular = batch['tabular'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, tabular)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                tabular = batch['tabular'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, tabular)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= threshold).float()
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        val_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Micro F1: {val_micro_f1:.4f}, Val Macro F1: {val_macro_f1:.4f}")
        
        # Save best model
        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            best_model_state = model.state_dict().copy()
            print(f"  New best model saved! (Val Micro F1: {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
