# Toxic Comment Classification with TensorFlow and Gradio

This project builds a machine learning pipeline to classify comments for various types of toxicity using a Bidirectional LSTM model. It includes data preprocessing, training, evaluation, and a user-friendly Gradio interface for prediction.

---

## Dataset
Kaggle : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Features

- Multi-label classification for six toxicity categories:
  - Toxic
  - Severe Toxic
  - Obscene
  - Threat
  - Insult
  - Identity Hate
- Preprocessing with TensorFlow's `TextVectorization`.
- Interactive Gradio interface for testing the model.
- Model evaluation with metrics such as Precision, Recall, and Accuracy.

---

![image alt](https://github.com/Lucifer1406/Deep_Learning_RNN_Based_Projects/blob/2bcd241a380364418a7fd604aa62500e26ed29ed/Comment_Toxicity/Screenshot%202025-01-03%20202705.png)


## Requirements

- Python 3.7+
- TensorFlow 2.0+
- Pandas
- Numpy
- Gradio
- Matplotlib

Install dependencies using:
pip install tensorflow pandas numpy gradio matplotlib

## Code Workflow

### 1. Data Preprocessing
- **Text Tokenization and Padding:**  
  Utilizes TensorFlow's `TextVectorization` to preprocess the input text by converting words into numerical tokens and padding sequences to a fixed length.
- **Data Pipeline Creation:**  
  Builds an efficient data pipeline using TensorFlow's `Dataset` API for caching, shuffling, batching, and prefetching data to improve performance.

### 2. Model Architecture
- **Embedding Layer:**  
  Converts input text into dense numerical vectors for the model to process.
- **Bidirectional LSTM Layer:**  
  Captures context from both forward and backward directions in the text sequence, enabling the model to understand relationships between words effectively.
- **Dense Layers:**  
  A series of fully connected layers to extract features and classify the input data.
- **Output Layer:**  
  Contains six neurons with sigmoid activation functions for multi-label classification of the toxicity categories.

### 3. Training and Evaluation
- **Model Compilation:**  
  The model is compiled using the Adam optimizer and binary cross-entropy loss for multi-label classification.
- **Data Splits:**  
  - **Training Set:** 70% of the dataset used to train the model.  
  - **Validation Set:** 20% of the dataset used for hyperparameter tuning and monitoring performance.  
  - **Test Set:** 10% of the dataset used to evaluate the final model.
- **Evaluation Metrics:**  
  The model's performance is evaluated using metrics like Precision, Recall, and Accuracy.

### 4. Gradio Interface
- **Interactive Interface:**  
  Implements a Gradio app to accept user inputs as text and display predictions for each of the six toxicity categories.
- **Real-time Prediction:**  
  Users can input any comment, and the app predicts the likelihood of each toxicity type in real-time.

--Compile the model using the Adam optimizer and binary cross-entropy loss.
--Train with 70% of data, validate with 20%, and test with the remaining 10%.
## Gradio Interface
--Implements a Gradio app to accept user input and display toxicity predictions.

#This file can now be directly saved as `README.md` and used for your project. Let me know if further customization is needed!
