# Neural-Network-SMS-Text-Classifier
Neural Network SMS Text Classifier  Machine Learning with Python
# Neural Network SMS Text Classifier

This project is a machine learning-based SMS text classifier that detects whether a given message is **spam** or **ham** (not spam). It is built using Python and TensorFlow, and it leverages a simple neural network with text vectorization and embeddings.

---

## 🚀 Project Overview

The classifier processes SMS messages, converts the text into numerical representations using TensorFlow’s `TextVectorization` layer, and trains a neural network to classify messages as spam or ham. The model is trained and validated on a labeled SMS dataset.

---

## 📂 Dataset

The project uses a publicly available SMS spam collection dataset provided by freeCodeCamp:

- **Training Data:** `train-data.tsv`
- **Validation Data:** `valid-data.tsv`

Each dataset contains two columns: `label` (spam/ham) and `message` (text).

---

## 🧰 Technologies

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy

---

## 💡 How It Works

1. **Data Loading:** Load and preprocess the SMS messages and labels.
2. **Text Vectorization:** Tokenize and integer-encode text messages with padding.
3. **Embedding:** Convert integer tokens to dense vector embeddings.
4. **Neural Network:** A feedforward neural network classifies messages with binary cross-entropy loss.
5. **Training:** The model is trained for 10 epochs on the training set and validated on the validation set.
6. **Prediction:** New SMS messages can be classified as spam or ham with a confidence score.

---

## ⚙️ Setup & Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/neural-network-sms-classifier.git
   cd neural-network-sms-classifier
