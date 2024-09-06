# NELEC Model for Contextual Emotion Detection in Text

## Overview

This repository contains the implementation of the **NELEC (Contextual Emotion Detection)** model for predicting emotions in textual data. The model leverages neural embeddings from GloVe and Emoji2Vec to handle text and emoji inputs, respectively. This project was developed as part of the third task in the SemEval-2019 competition, focusing on **'Contextual Emotion Detection in Text.'**

The architecture of the model includes **LSTM** (Long Short-Term Memory), **GRU** (Gated Recurrent Unit), and attention layers. It uses a combination of max-pooling and average-pooling to regulate the model, enhancing its performance. The implementation achieves an F1 score of **68%** on the task.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Results](#training-and-results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [References](#references)

## Dataset

The dataset for the **NELEC model** is provided by Microsoft as part of the **EmoContext** task. It contains labeled conversations, including both text and emoji data. Each conversation is classified into one of four categories:
- **Happy**
- **Sad**
- **Angry**
- **Others**

The dataset is split into training, testing, and development sets, each with a distribution of these emotion labels. A table in the report outlines the global statistics of the dataset, including train, test, and dev data proportions.

## Preprocessing

### 1. **Text Cleaning:**
   - Conversations are cleaned using regular expressions to revert abbreviations and typographical errors to their full forms.
   - Lamentation and normalization steps, such as the removal of stop words and punctuation, are skipped based on previous research findings.

### 2. **Embeddings:**
   - **GloVe (Global Vectors for Word Representation):** Pre-trained embeddings are used for words, providing a 300-dimensional vector for each word.
   - **Emoji2Vec:** Emojis are represented using the Emoji2Vec embeddings, which are trained to capture the meanings of emojis in a similar manner as words.

## Model Architecture

The model uses a combination of:
- **LSTM (Long Short-Term Memory)** layers for capturing long-term dependencies in the input sequences.
- **GRU (Gated Recurrent Unit)** layers to handle sequential data with fewer parameters compared to LSTM.
- **Attention Mechanism:** A simplified version of Bahdanau's attention is employed to improve focus on important tokens in the input.
- **Pooling Layers:** Max-pooling and average-pooling layers are used to regulate the output of the LSTM and GRU layers.

A detailed diagram of the model architecture is provided in the report, explaining the flow of data through the embedding, LSTM/GRU, and pooling layers.

## Training and Results

### 1. **Training Procedure:**
   - The model is trained on a dataset of 24,128 conversations using **Pytorch** with 300-dimensional GloVe embeddings and a sequence length of 35 tokens.
   - The model is trained for 100 epochs using a cyclic learning rate.

### 2. **Evaluation:**
   - Precision, Recall, and F1-scores are calculated for each class (Happy, Sad, Angry, Others) and are compared between the original Keras model, a retrained Keras model, and the implemented Pytorch model.

### 3. **Results:**
   - The final model achieves an F1-score of **68%**.
   - Detailed class-wise performance (Happy, Sad, Angry, Others) and macro/micro average scores are provided.

## Requirements

- **Python 3.7+**
- **Pytorch**
- **Keras (for comparison)**
- **Numpy**
- **Pandas**
- **GloVe embeddings** (downloaded separately)
- **Emoji2Vec embeddings**

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nelec-emotion-detection.git
   cd nelec-emotion-detection
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the GloVe and Emoji2Vec embeddings and place them in the appropriate directories.

4. Run the training script:
   ```bash
   python train.py --epochs 100 --embedding glove --data_path ./data/
   ```

## Conclusion

The NELEC model demonstrates strong performance in detecting emotions from contextual conversations, using both word and emoji embeddings. The incorporation of LSTM, GRU, and attention mechanisms allows the model to capture important long-term dependencies in the text. The project explores possible improvements in handling the "Others" class, where class imbalance is an issue.

## References

- Agrawal, Suri (2019). "Contextual Emotion Detection in Text." SemEval-2019 Task.
- Pennington, Socher, Manning (2014). "GloVe: Global Vectors for Word Representation."
- Raffel, Ellis (2016). "Feed-forward Attention Models."
