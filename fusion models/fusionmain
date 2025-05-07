"""
Modularized Multimodal Fusion Model
-----------------------------------
A modular implementation of a trimodal fusion model with image, text, and tabular data.
This module is designed to be integrated into larger multimodal systems.
"""

import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, classification_report
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# Ensure PIL can load truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------------------------
# Configuration Module
# ------------------------------------------------

class ModelConfig:
    """Configuration class for model parameters"""
    
    def __init__(self, **kwargs):
        # Model architecture
        self.bert_model_name = kwargs.get('bert_model_name', 'bert-base-uncased')
        self.image_model_name = kwargs.get('image_model_name', 'efficientnet_b0')
        self.max_len = kwargs.get('max_len', 128)
        self.image_size = kwargs.get('image_size', 224)
        
        # Training parameters
        self.batch_size = kwargs.get('batch_size', 16)
        self.epochs = kwargs.get('epochs', 20)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.weight_decay = kwargs.get('weight_decay', 0.015)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 2)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 2)
        
        # Loss function parameters
        self.use_focal_loss = kwargs.get('use_focal_loss', True)
        self.focal_loss_gamma = kwargs.get('focal_loss_gamma', 2.0)
        self.threshold = kwargs.get('threshold', 0.4)
        self.label_smoothing_factor = kwargs.get('label_smoothing_factor', 0.05)
        
        # Dropout rates
        self.dropout_image = kwargs.get('dropout_image', 0.4)
        self.dropout_text = kwargs.get('dropout_text', 0.4)
        self.dropout_tabular_mlp1 = kwargs.get('dropout_tabular_mlp1', 0.4)
        self.dropout_tabular_mlp2 = kwargs.get('dropout_tabular_mlp2', 0.3)
        self.dropout_fusion = kwargs.get('dropout_fusion', 0.5)
        
        # File paths
        self.model_save_path = kwargs.get('model_save_path', 'best_trimodal_model.pth')
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# Dataset Module
# ------------------------------------------------

class MultimodalDataset(Dataset):
    """Dataset for handling multiple modalities: image, text, and tabular data"""
    
    def __init__(self, image_paths: List[str], encodings: Dict, tabular_feats: np.ndarray, 
                 labels: np.ndarray, transforms: transforms.Compose):
        self.image_paths = image_paths
        self.encodings = encodings
        self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['tabular_features'] = self.tabular_feats[idx].clone().detach()
        item['labels'] = self.labels[idx].clone().detach()
        
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            item['image'] = self.transforms(image)
        except Exception:
            # Return zero tensor if image loading fails
            item['image'] = torch.zeros((3, 224, 224), dtype=torch.float32)
        return item

    def __len__(self) -> int:
        return len(self.labels)

# ------------------------------------------------
# Model Components
# ------------------------------------------------

def get_image_model(model_name: str = 'efficientnet_b0', pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Initialize and configure an image backbone model"""
    
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        img_model = models.resnet50(weights=weights)
        num_ftrs = img_model.fc.in_features
        img_model.fc = nn.Identity()
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        img_model = models.efficientnet_b0(weights=weights)
        num_ftrs = img_model.classifier[1].in_features
        img_model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported image model: {model_name}")

    if pretrained:
        # Freeze all parameters
        for param in img_model.parameters():
            param.requires_grad = False
            
        # Selectively unfreeze specific layers
        if model_name.startswith('resnet'):
            # Unfreeze only the last block of layer4
            for param in img_model.layer4[-1].parameters():
                param.requires_grad = True
        elif model_name.startswith('efficientnet'):
            # Unfreeze last 2 blocks
            for param in img_model.features[-2:].parameters():
                param.requires_grad = True
                
    return img_model, num_ftrs

class TabularEncoder(nn.Module):
    """Encoder for tabular features"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, 
                 dropout1: float = 0.4, dropout2: float = 0.3):
        super(TabularEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TextEncoder(nn.Module):
    """Encoder for text data using BERT"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.4):
        super(TextEncoder, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Freeze all BERT parameters initially
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Unfreeze only the last BERT layer and pooler
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.pooler_output)

class FocalLoss(nn.Module):
    """Focal Loss implementation for multi-label classification"""
    
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        epsilon = 1e-8  # To prevent log(0)
        
        # Calculate pt for numerical stability
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1.0 - pt + epsilon) ** self.gamma
        loss = focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------------------------------------
# Main Fusion Model
# ------------------------------------------------

class MultimodalFusionModel(nn.Module):
    """
    Fusion model integrating image, text, and tabular data for multi-label classification
    """
    
    def __init__(self, config: ModelConfig, num_labels: int, tabular_input_dim: int):
        super(MultimodalFusionModel, self).__init__()
        
        # Image encoder
        self.image_model, image_feature_dim = get_image_model(
            config.image_model_name, pretrained=True)
        self.image_dropout = nn.Dropout(config.dropout_image)
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=config.bert_model_name, 
            dropout=config.dropout_text)
        text_feature_dim = self.text_encoder.bert.config.hidden_size
        
        # Tabular encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            hidden_dim=96,
            dropout1=config.dropout_tabular_mlp1, 
            dropout2=config.dropout_tabular_mlp2)
        tabular_feature_dim = 96  # matches hidden_dim in TabularEncoder
        
        # Feature fusion and classification
        combined_dim = image_feature_dim + text_feature_dim + tabular_feature_dim
        self.fusion_dropout = nn.Dropout(config.dropout_fusion)
        self.classifier = nn.Linear(combined_dim, num_labels)

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, tabular_features: torch.Tensor) -> torch.Tensor:
        # Process each modality
        image_features = self.image_model(image)
        image_features = self.image_dropout(image_features)
        
        text_features = self.text_encoder(input_ids, attention_mask)
        
        tabular_features = self.tabular_encoder(tabular_features)
        
        # Fusion
        combined_features = torch.cat(
            (image_features, text_features, tabular_features), dim=1)
        fused_features = self.fusion_dropout(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        return logits

# ------------------------------------------------
# Trainer Module
# ------------------------------------------------

class MultimodalTrainer:
    """Trainer class for handling the training and evaluation process"""
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        
        # Setup loss function
        self.setup_loss_function()
        
        # Setup optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
    
    def setup_loss_function(self, pos_weight: Optional[torch.Tensor] = None):
        """Setup loss function based on configuration"""
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
            
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(
                gamma=self.config.focal_loss_gamma, 
                pos_weight=pos_weight, 
                reduction='mean')
            print("Using Focal Loss")
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print("Using BCEWithLogitsLoss with pos_weight")
    
    def setup_optimizer_and_scheduler(self, train_loader):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        total_steps = (len(train_loader) // self.config.gradient_accumulation_steps) * self.config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(0.1 * total_steps), 
            num_training_steps=total_steps
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch"""
        self.model.train()
        total_train_loss = 0
        train_preds_list = []
        train_labels_list = []
        
        train_loop = tqdm(enumerate(train_loader), 
                          desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]", 
                          total=len(train_loader), 
                          leave=False)
        
        for batch_idx, batch in train_loop:
            # Move data to device
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            tabular = batch['tabular_features'].to(self.device, non_blocking=True)
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Apply label smoothing if enabled and not using focal loss
            if not self.config.use_focal_loss and self.config.label_smoothing_factor > 0:
                num_labels = labels.size(1)
                labels = labels * (1.0 - self.config.label_smoothing_factor) + self.config.label_smoothing_factor / num_labels
            
            # Forward pass
            outputs = self.model(
                image=images, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                tabular_features=tabular
            )
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            loss = loss / self.config.gradient_accumulation_steps  # Normalize loss
            
            # Backward pass
            loss.backward()
            
            # Update weights on accumulation step
            if ((batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or 
                (batch_idx + 1) == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_train_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Calculate predictions
            preds = (torch.sigmoid(outputs) >= self.config.threshold).int().cpu().numpy()
            train_preds_list.append(preds)
            train_labels_list.append(labels.cpu().numpy())
            
            train_loop.set_postfix(loss=loss.item() * self.config.gradient_accumulation_steps)
        
        # Calculate metrics
        metrics = {}
        metrics['loss'] = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        if train_preds_list:
            train_preds_all = np.vstack(train_preds_list)
            train_labels_all = np.vstack(train_labels_list)
            metrics['micro_f1'] = f1_score(train_labels_all, train_preds_all, average='micro', zero_division=0)
            metrics['macro_f1'] = f1_score(train_labels_all, train_preds_all, average='macro', zero_division=0)
        
        return metrics
    
    def evaluate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Evaluate the model on validation data"""
        self.model.eval()
        total_val_loss = 0
        val_preds_list = []
        val_labels_list = []
        
        val_loop = tqdm(val_loader, 
                        desc=f"Epoch {epoch+1}/{self.config.epochs} [Val]", 
                        leave=False)
        
        with torch.no_grad():
            for batch in val_loop:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                tabular = batch['tabular_features'].to(self.device, non_blocking=True)
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(
                    image=images, 
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    tabular_features=tabular
                )
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Calculate predictions
                preds = (torch.sigmoid(outputs) >= self.config.threshold).int().cpu().numpy()
                val_preds_list.append(preds)
                val_labels_list.append(labels.cpu().numpy())
                
                val_loop.set_postfix(loss=loss.item())
        
        # Calculate metrics
        metrics = {}
        metrics['loss'] = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        if val_preds_list:
            val_preds_all = np.vstack(val_preds_list)
            val_labels_all = np.vstack(val_labels_list)
            metrics['micro_f1'] = f1_score(val_labels_all, val_preds_all, average='micro', zero_division=0)
            metrics['macro_f1'] = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)
            metrics['hamming_loss'] = hamming_loss(val_labels_all, val_preds_all)
            metrics['accuracy'] = accuracy_score(val_labels_all, val_preds_all)
            metrics['preds'] = val_preds_all
            metrics['labels'] = val_labels_all
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Main training loop with early stopping"""
        if self.optimizer is None:
            self.setup_optimizer_and_scheduler(train_loader)
        
        best_val_f1 = 0.0
        epochs_no_improve = 0
        training_history = {'train': [], 'val': []}
        
        for epoch in range(self.config.epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            training_history['train'].append(train_metrics)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader, epoch)
            training_history['val'].append(val_metrics)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{self.config.epochs} -> "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Micro F1: {train_metrics.get('micro_f1', 0):.4f}, "
                  f"Train Macro F1: {train_metrics.get('macro_f1', 0):.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Micro F1: {val_metrics.get('micro_f1', 0):.4f}, "
                  f"Val Macro F1: {val_metrics.get('macro_f1', 0):.4f}")
            
            # Save best model
            val_micro_f1 = val_metrics.get('micro_f1', 0)
            if val_micro_f1 > best_val_f1:
                best_val_f1 = val_micro_f1
                torch.save(self.model.state_dict(), self.config.model_save_path)
                print(f"New best model saved with Val Micro F1: {best_val_f1:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Val Micro F1 did not improve for {epochs_no_improve} epoch(s). Best: {best_val_f1:.4f}")
            
            # Early stopping
            if epochs_no_improve >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
            # Memory cleanup
            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
        
        return {'best_val_f1': best_val_f1, 'history': training_history}
    
    def final_evaluation(self, val_loader: DataLoader, class_names: List[str]) -> Dict:
        """Perform final evaluation using the best saved model"""
        print(f"\n--- Loading best model from {self.config.model_save_path} for final evaluation ---")
        
        if os.path.exists(self.config.model_save_path):
            # Load the best model
            self.model.load_state_dict(torch.load(self.config.model_save_path, map_location=self.device))
            print("Best model weights loaded for final evaluation.")
        else:
            print(f"Warning: Model file {self.config.model_save_path} not found. "
                  f"Evaluating with the current model state.")
        
        self.model.eval()
        
        # Evaluate with best model
        metrics = self.evaluate(val_loader, 0)
        
        if 'preds' in metrics and 'labels' in metrics:
            # Detailed classification report
            y_pred = metrics['preds']
            y_true = metrics['labels']
            report = classification_report(y_true, y_pred, 
                                          target_names=class_names, 
                                          zero_division=0)
            
            print("\nFinal Validation Metrics:")
            print(f"Micro F1 Score: {metrics.get('micro_f1', 0):.4f}")
            print(f"Macro F1 Score: {metrics.get('macro_f1', 0):.4f}")
            print(f"Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
            print(f"Subset Accuracy: {metrics.get('accuracy', 0):.4f}")
            print("\nClassification Report (Validation Set - best model):")
            print(report)
            
            metrics['classification_report'] = report
        
        return metrics

# ------------------------------------------------
# Utility Functions
# ------------------------------------------------

def create_image_transforms(image_size: int, is_training: bool = True) -> transforms.Compose:
    """
    Create appropriate image transformations based on training/evaluation mode
    
    Args:
        image_size: Target image size
        is_training: Whether to include data augmentation for training
        
    Returns:
        Composition of image transformations
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(
    config: ModelConfig,
    X_train_texts: List[str],
    X_val_texts: List[str],
    X_train_tabular: np.ndarray,
    X_val_tabular: np.ndarray,
    train_image_paths: List[str],
    val_image_paths: List[str],
    y_train: np.ndarray,
    y_val: np.ndarray
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for train and validation sets
    
    Args:
        config: Configuration object
        X_train_texts: List of training text data
        X_val_texts: List of validation text data
        X_train_tabular: Tabular features for training
        X_val_tabular: Tabular features for validation
        train_image_paths: Paths to training images
        val_image_paths: Paths to validation images
        y_train: Training labels
        y_val: Validation labels
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    
    # Tokenize texts
    train_encodings = tokenizer(X_train_texts, padding=True, truncation=True, 
                               max_length=config.max_len, return_tensors='pt')
    val_encodings = tokenizer(X_val_texts, padding=True, truncation=True, 
                             max_length=config.max_len, return_tensors='pt')
    
    # Create image transforms
    train_transforms = create_image_transforms(config.image_size, is_training=True)
    val_transforms = create_image_transforms(config.image_size, is_training=False)
    
    # Create datasets
    train_dataset = MultimodalDataset(train_image_paths, train_encodings, 
                                     X_train_tabular, y_train, train_transforms)
    val_dataset = MultimodalDataset(val_image_paths, val_encodings, 
                                   X_val_tabular, y_val, val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        persistent_workers=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


  

if __name__ == "__main__":
    example_usage()
