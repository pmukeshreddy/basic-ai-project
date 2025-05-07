# Trimodal Movie Genre Classification System

## Overview
This project implements a comprehensive movie genre classification system that leverages three distinct data modalities:
- **Text**: Movie titles and overviews
- **Images**: Movie poster visuals
- **Tabular Data**: Metadata like release dates, vote averages, etc.

By combining these three modalities, we achieve enhanced classification performance compared to using any single data source alone.

## Project Structure

### 1. Textual Processing Component
- **Genre Analysis**: Transformation of raw genre strings into Python lists for frequency analysis
- **Genre Grouping**: Categorization of movies into broader thematic groups based on individual genres
- **Feature Engineering**:
  - Sentiment polarity of movie titles
  - Sequel identification through title analysis
  - Genre-specific keyword counting in overviews
  - Sentiment analysis of overviews (polarity and subjectivity)
  - Release season extraction and one-hot encoding
- **Preprocessing Pipeline**:
  - Missing value handling
  - Duplicate removal
  - Text normalization (lowercase conversion, special character removal)
  - Tokenization and stop word filtering
  - Word stemming
  - Feature standardization

### 2. Vision Processing Component
- **Image Preprocessing**: 
  - Format standardization
  - Resizing
  - Normalization
- **Data Augmentation**:
  - Random rotations
  - Horizontal flips
  - Color jittering
- **Feature Extraction**: Implementation of pre-trained EfficientNet for balance between computational efficiency and accuracy

### 3. Fusion Component
- **Configuration**: Environment setup with GPU acceleration and parameter definition
- **Model Architecture**:
  - Image pathway: EfficientNet
  - Text pathway: BERT
  - Tabular pathway: Multi-layer perceptron
  - Fusion layer: Combined representation for final classification
- **Training Process**:
  - Class imbalance handling through Focal Loss or weighted BCE loss with label smoothing
  - Multi-epoch training with gradient accumulation
  - Early stopping based on validation performance
  - Performance tracking (micro/macro F1 scores)
  - Model saving based on validation micro F1 score

## Results
The trimodal approach demonstrates significant improvements over single-modality baselines, with detailed performance metrics including:
- F1 scores (micro and macro)
- Hamming loss
- Classification reports for each target group

## Requirements
- Llama (for text processing)
- vision transformer (for image processing)
- PyTorch (for deep learning implementation)
- BERT (for text feature extraction)
- Standard data science libraries (pandas, numpy, scikit-learn)
- GPU support recommended for efficient training


