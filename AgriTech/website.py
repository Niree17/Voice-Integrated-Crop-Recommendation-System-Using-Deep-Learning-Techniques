import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import base64
import re
from gtts import gTTS
import speech_recognition as sr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import groq

# Initialize Groq client
groq_client = groq.Client(api_key="YOUR_GROQ_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="🌾 AgriTech Crop Recommendation", page_icon="🌱", layout="wide")

# ================================
# Text-to-Speech
# ================================
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = "output.mp3"
    tts.save(audio_file)
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'

# ================================
# Voice Input
# ================================
def voice_input():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("🎙️ Listening... Please speak now.")
            audio = r.listen(source)
            st.write("Processing your speech...")
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand your speech.")
    except sr.RequestError:
        st.error("Could not request results from speech recognition service.")
    except Exception:
        st.error("Could not access microphone. Please check your mic settings.")
    return None

# ================================
# Parse Voice Input
# ================================
def parse_voice_input(text):
    text = text.lower()
    params = {}
    for key in ["ph", "nitrogen", "potassium", "temperature", "rainfall"]:
        if key in text:
            words = text.split()
            for i, word in enumerate(words):
                if word == key and i + 1 < len(words):
                    val = re.sub("[^0-9.]", "", words[i + 1])
                    if val.replace(".", "", 1).isdigit():
                        params[key] = float(val)
    return params

# ================================
# Crop Prediction with DL Model + Groq Explanation
# ================================
def predict_with_model_and_groq(input_data, model_name):
    ph, n, k20, temperature, rainfall = input_data

    df = pd.read_excel("dataset.xlsx")
    X = df[["PH", "N", "K20", "Temparature", "Rainfall"]]
    y = df["CROP"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Avoid stratify error
    unique, counts = np.unique(y_enc, return_counts=True)
    if np.min(counts) < 2:
        st.warning("⚠️ Some crops have only one record — training without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

    # Build model
    model = Sequential()
    if model_name == "LSTM":
        model.add(LSTM(64, input_shape=(1, 5)))
    elif model_name == "BiLSTM":
        model.add(Bidirectional(LSTM(64, input_shape=(1, 5))))
    else:
        model.add(GRU(64, input_shape=(1, 5)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(len(np.unique(y_enc)), activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model.fit(X_train_reshaped, y_train, epochs=8, batch_size=16, verbose=0)

    # Predict
    input_scaled = scaler.transform([[ph, n, k20, temperature, rainfall]])
    input_reshaped = np.reshape(input_scaled, (1, 1, 5))
    prediction = model.predict(input_reshaped)
    crop_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)
    predicted_crop = le.inverse_transform([crop_index])[0]

    # Explain with Groq
    try:
        explanation_prompt = f"""
        You are an agricultural expert. Explain briefly why {predicted_crop} is suitable 
        for soil with pH={ph}, N={n}, K2O={k20}, temperature={temperature}°C, and rainfall={rainfall}mm.
        Respond in 2–3 sentences, concise and factual.
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": explanation_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=100,
        )
        explanation = chat_completion.choices[0].message.content.strip()
    except Exception:
        explanation = "Explanation unavailable due to network or API issue."

    return predicted_crop, confidence, explanation

# ================================
# Streamlit App
# ================================
def main():
    st.title("🌾 AgriTech Crop Recommendation System")
    st.write("Get intelligent crop recommendations based on soil and weather using deep learning + Groq AI.")

    model_name = st.sidebar.radio("Choose a Model", ["LSTM", "BiLSTM", "GRU"])
    input_method = st.sidebar.radio("Choose Input Method", ["Text Input", "Voice Input"])
    output_method = st.sidebar.multiselect("Output Format", ["Text Output", "Voice Output"], default=["Text Output"])

    # Default params
    if "params" not in st.session_state:
        st.session_state.params = {"ph": 6.5, "nitrogen": 60.0, "potassium": 200.0, "temperature": 28.0, "rainfall": 120.0}

    # Voice input section
    if input_method == "Voice Input":
        st.subheader("🎤 Voice Input")
        if st.button("Start Voice Input"):
            voice_text = voice_input()
            if voice_text:
                st.success(f"You said: {voice_text}")
                parsed = parse_voice_input(voice_text)
                if parsed:
                    for k, v in parsed.items():
                        st.session_state.params[k] = v
                    st.success("✅ Parameters updated from voice input!")

    # Parameter sliders
    st.subheader("🧪 Enter Soil and Climate Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.params["ph"] = st.slider("pH", 0.0, 14.0, st.session_state.params["ph"], 0.1)
        st.session_state.params["nitrogen"] = st.slider("Nitrogen (N)", 0.0, 200.0, st.session_state.params["nitrogen"], 1.0)
    with col2:
        st.session_state.params["potassium"] = st.slider("Potassium (K2O)", 0.0, 500.0, st.session_state.params["potassium"], 1.0)
        st.session_state.params["temperature"] = st.slider("Temperature (°C)", 0.0, 50.0, st.session_state.params["temperature"], 0.1)
        st.session_state.params["rainfall"] = st.slider("Rainfall (mm)", 0.0, 3000.0, st.session_state.params["rainfall"], 10.0)

    # Predict button
    if st.button("🔍 Get Crop Recommendation"):
        with st.spinner("Analyzing soil and weather conditions..."):
            input_data = [
                st.session_state.params["ph"],
                st.session_state.params["nitrogen"],
                st.session_state.params["potassium"],
                st.session_state.params["temperature"],
                st.session_state.params["rainfall"],
            ]
            crop, conf, explain = predict_with_model_and_groq(input_data, model_name)

            st.subheader("🌱 Prediction Result")
            st.success(f"**Recommended Crop:** {crop}")
            st.info(f"**Confidence:** {conf:.2f}%")
            st.markdown(f"**🧠 Groq Explanation:** {explain}")

            if "Voice Output" in output_method:
                voice_text = f"Based on your soil and climate data, the best crop to grow is {crop}, with a confidence of {conf:.0f} percent."
                st.markdown(text_to_speech(voice_text), unsafe_allow_html=True)

    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "This app combines deep learning (LSTM/BiLSTM/GRU) with Groq AI to recommend crops "
        "and provide smart agricultural insights. Supports both text and voice interaction."
    )

if __name__ == "__main__":
    main()
