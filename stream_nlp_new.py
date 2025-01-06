import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
import nltk

# Ensure nltk data is available
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download stopwords if not already available
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
    print("Stopwords found.")
except LookupError:
    print("Stopwords not found. Downloading...")
    nltk.download('stopwords', download_dir=nltk_data_path)

# Define custom metrics
@tf.keras.utils.register_keras_serializable()
def precision_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.round(tf.clip_by_value(y_pred, 0, 1)), tf.float32))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

@tf.keras.utils.register_keras_serializable()
def recall_m(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(tf.round(tf.clip_by_value(y_true, 0, 1)), tf.float32))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

@tf.keras.utils.register_keras_serializable()
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model(
    'CNN_model_trial4.h5',
    custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m}
)

# Preprocessing functions
def casefolding(text):
    import re
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r"\s*#\S+", "", text)
    text = re.sub(r"\s*@\S+", "", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.strip()
    return text

def remove_stop_word(text):
    from nltk.corpus import stopwords
    stopwords_eng = stopwords.words('english')
    words = text.split()
    return " ".join([word for word in words if word not in stopwords_eng])

def preprocess_input_text(text):
    text = casefolding(text)
    text = remove_stop_word(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=50)  # Adjust maxlen based on your model
    return padded_sequence

# Predict sentiment
def predict_sentiment(input_text):
    processed_input = preprocess_input_text(input_text)
    prediction = model.predict(processed_input)
    sentiment = np.argmax(prediction, axis=1)[0]  # 0, 1, or 2
    confidence = prediction[0][sentiment]
    if sentiment == 0:
        return "bad", confidence
    elif sentiment == 1:
        return "good", confidence
    else:
        return "neutral", confidence

# Streamlit UI
st.title("Analisis Sentiment Menggunakan CNN")
st.write("Masukkan Kalimat Untuk Prediksi Sentimen.")

# Input text
user_input = st.text_area("Masukkan Kalimatnya:", "")
if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Klik Enter!")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
