import pandas as pd
import csv
import os
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

# --- Configuration ---
CSV_FILE_PATH = "/kaggle/input/basics-of-ai/movie_dataset_complete/movies_data.csv"
IMAGE_FOLDER_PATH = "/kaggle/input/basics-of-ai/movie_dataset_complete/posters"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Training Hyperparameters
BATCH_SIZE = 8  # Consistent with the training loop in the notebook
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
TRAIN_SPLIT_RATIO = 0.8
IMAGE_SIZE = (224, 224) # ViT standard size

# --- Data Loading and Preprocessing ---

def load_image_paths_and_labels(csv_path, image_folder):
    """Loads image paths and their corresponding genre labels from a CSV file."""
    image_label_list = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_filename = row['id'] + '.jpg'
            full_image_path = os.path.join(image_folder, image_filename)
            label = row['genres'] # This is a string representation of a list
            image_label_list.append((full_image_path, label))
    print(f"Loaded {len(image_label_list)} image-label pairs.")
    return image_label_list

def get_multilabel_binarizer(image_label_list):
    """Fits and returns a MultiLabelBinarizer on the genre lists."""
    genre_lists = [ast.literal_eval(label_str) for _, label_str in image_label_list]
    mlb = MultiLabelBinarizer()
    mlb.fit(genre_lists)
    print("Fitted MultiLabelBinarizer. Genre classes:", mlb.classes_)
    return mlb

# --- PyTorch Dataset ---

class MoviePosterDataset(Dataset):
    """Custom Dataset for movie posters and their multi-label genres."""
    def __init__(self, data_list, mlb, transform=None):
        self.data_list = data_list
        self.mlb = mlb
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label_str = self.data_list[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            genres = ast.literal_eval(label_str)
            # Transform genres into a multi-hot encoded vector
            label_vec = torch.tensor(self.mlb.transform([genres])[0], dtype=torch.float32)
            
            return image, label_vec
        except (FileNotFoundError, OSError, SyntaxError) as e:
            # print(f"Warning: Skipping {image_path} due to error: {e}. Returning next valid sample.")
            # Recursively try the next item, ensuring not to go into an infinite loop if all fail
            return self.__getitem__((idx + 1) % len(self.data_list))


def get_data_loaders(image_label_list, mlb, batch_size, train_split_ratio, image_size_tuple):
    """Creates and returns train and validation DataLoaders."""
    transform = transforms.Compose([
        transforms.Resize(image_size_tuple),
        transforms.ToTensor(),
        # ViT models usually have their own normalization, often handled by ViTFeatureExtractor
        # If using ViTFeatureExtractor, normalization might be applied there or you can add it here:
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Example for some ViT models
        # For 'google/vit-base-patch16-224-in21k', HuggingFace handles it if pixel_values are 0-1.
    ])

    dataset = MoviePosterDataset(data_list=image_label_list, mlb=mlb, transform=transform)
    
    train_size = int(train_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # Ensure reproducible splits if needed by setting torch.manual_seed
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Created DataLoaders: Train size={len(train_dataset)}, Val size={len(val_dataset)}")
    return train_loader, val_loader

# --- Model Definition ---

class ViTForMultiLabelClassification(nn.Module):
    """Vision Transformer model for multi-label image classification."""
    def __init__(self, num_labels, model_name=VIT_MODEL_NAME):
        super(ViTForMultiLabelClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        # The ViTFeatureExtractor associated with the model might be useful for preprocessing
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        
        in_features = self.vit.config.hidden_size # e.g., 768 for vit-base
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
            nn.Sigmoid() 
        )

    def forward(self, pixel_values):
        # If using ViTFeatureExtractor outside, pixel_values would be preprocessed.
        # Otherwise, ensure pixel_values are [B, C, H, W] and appropriately normalized.
        outputs = self.vit(pixel_values=pixel_values)
        features = outputs.pooler_output # [B, hidden_size]
        logits = self.classifier(features)
        return logits

# --- Training Utilities ---

def get_class_weights(mlb_classes, device):
    """
    Calculates class weights for weighted loss function.
    Uses a predefined dictionary of weights.
    """
    class_weights_dict = {
        'Family': 10.772090188909202, 'Comedy': 3.4145257871354064,
        'Adventure': 7.08496993987976, 'Fantasy': 10.404355503237198,
        'Action': 4.92670011148272, 'Crime': 7.75647213690215,
        'Thriller': 4.620230005227391, 'Science Fiction': 10.118488838008014,
        'Mystery': 12.735590778097983, 'Romance': 6.202456140350877,
        'Drama': 2.3311354345245947, 'Horror': 7.256568144499179,
        'War': 18.204943357363543, 'Animation': 11.027448533998752,
        'Music': 18.865528281750265, 'History': 15.39808362369338,
        'Documentary': 17.212268743914315, 'TV Movie': 57.20711974110032,
        'Western': 25.730713245997087
    }
    # Ensure mlb_classes is a list of strings for consistent ordering
    ordered_classes = list(mlb_classes)
    
    # Check if all classes in mlb_classes are in class_weights_dict
    for cls in ordered_classes:
        if cls not in class_weights_dict:
            print(f"Warning: Class '{cls}' from dataset not found in predefined class_weights_dict. Using weight 1.0.")
            class_weights_dict[cls] = 1.0 # Default weight if missing

    weights_list = [class_weights_dict[cls] for cls in ordered_classes]
    pos_weight_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
    print("Positive class weights for BCELoss:", pos_weight_tensor)
    return pos_weight_tensor

def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch_num, total_epochs):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds_epoch = []
    all_targets_epoch = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Training]", dynamic_ncols=True)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Store predictions and targets for F1 score calculation
        preds_batch = (outputs.detach().cpu().numpy() > 0.5).astype(int)
        targets_batch = labels.cpu().numpy().astype(int)
        all_preds_epoch.append(preds_batch)
        all_targets_epoch.append(targets_batch)

        # Cumulative F1 for progress bar (optional, can be slow)
        # flat_preds_so_far = np.vstack(all_preds_epoch)
        # flat_targets_so_far = np.vstack(all_targets_epoch)
        # micro_f1_so_far = f1_score(flat_targets_so_far, flat_preds_so_far, average="micro", zero_division=0)
        
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            # "micro_f1_batch": f"{micro_f1_so_far:.4f}" 
        })

    avg_epoch_loss = total_loss / len(data_loader)
    
    # Calculate F1 score for the entire epoch
    final_preds_epoch = np.vstack(all_preds_epoch)
    final_targets_epoch = np.vstack(all_targets_epoch)
    epoch_micro_f1 = f1_score(final_targets_epoch, final_preds_epoch, average="micro", zero_division=0)
    
    print(f"\nEpoch {epoch_num+1} Training | Avg Loss: {avg_epoch_loss:.4f} | Micro F1: {epoch_micro_f1:.4f}")
    return avg_epoch_loss, epoch_micro_f1


def evaluate_model(model, data_loader, criterion, device, epoch_num, total_epochs):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    all_preds_epoch = []
    all_targets_epoch = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch_num+1}/{total_epochs} [Validation]", dynamic_ncols=True)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds_batch = (outputs.cpu().numpy() > 0.5).astype(int)
            targets_batch = labels.cpu().numpy().astype(int)
            all_preds_epoch.append(preds_batch)
            all_targets_epoch.append(targets_batch)
            
            # progress_bar.set_postfix({"loss": f"{loss.item():.4f}"}) # Optional: batch loss for val

    avg_epoch_loss = total_loss / len(data_loader)
    
    final_preds_epoch = np.vstack(all_preds_epoch)
    final_targets_epoch = np.vstack(all_targets_epoch)
    epoch_micro_f1 = f1_score(final_targets_epoch, final_preds_epoch, average="micro", zero_division=0)

    print(f"Epoch {epoch_num+1} Validation | Avg Loss: {avg_epoch_loss:.4f} | Micro F1: {epoch_micro_f1:.4f}\n")
    return avg_epoch_loss, epoch_micro_f1

# --- Main Execution ---

def main():
    """Main function to orchestrate the training and evaluation process."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and preprocess data
    image_label_list = load_image_paths_and_labels(CSV_FILE_PATH, IMAGE_FOLDER_PATH)
    if not image_label_list:
        print("No data loaded. Exiting.")
        return
        
    mlb = get_multilabel_binarizer(image_label_list)
    genre_classes = mlb.classes_
    num_labels = len(genre_classes)

    # 2. Create DataLoaders
    train_loader, val_loader = get_data_loaders(
        image_label_list, mlb, BATCH_SIZE, TRAIN_SPLIT_RATIO, IMAGE_SIZE
    )

    # 3. Initialize Model
    model = ViTForMultiLabelClassification(num_labels=num_labels, model_name=VIT_MODEL_NAME)
    model.to(device)

    # 4. Define Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Get class weights and move to device
    pos_weights = get_class_weights(genre_classes, device)
    criterion = nn.BCELoss(weight=pos_weights) # pos_weights are on the correct device

    # 5. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )
        val_loss, val_f1 = evaluate_model(
            model, val_loader, criterion, device, epoch, NUM_EPOCHS
        )
        # Potentially add model saving logic here based on val_f1 or val_loss
        print(f"--- End of Epoch {epoch+1}/{NUM_EPOCHS} ---")
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        print("-" * 30)

    print("Training complete.")
