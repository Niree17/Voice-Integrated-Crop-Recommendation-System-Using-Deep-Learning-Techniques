import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import base64
from gtts import gTTS
import speech_recognition as sr
from keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="AgriTech Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Load models and preprocessing tools
@st.cache_resource
def load_resources():
    models = {
        'LSTM': load_model('saved_models/LSTM.keras'),
        'BiLSTM': load_model('saved_models/BiLSTM.keras'),
        'GRU': load_model('saved_models/GRU.keras')
    }
    label_encoder = joblib.load('saved_models/label_encoder.joblib')
    scaler = joblib.load('saved_models/scaler.joblib')
    return models, label_encoder, scaler

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = "output.mp3"
    tts.save(audio_file)
    
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
    
    return audio_tag

# Function to capture voice input
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak now.")
        audio = r.listen(source)
        st.write("Processing your speech...")
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand your speech.")
        return None
    except sr.RequestError:
        st.error("Could not request results from speech recognition service.")
        return None

# Function to parse voice input for parameters
def parse_voice_input(text):
    text = text.lower()
    params = {}
    
    # Extract parameters from voice input
    if "ph" in text:
        words = text.split()
        for i, word in enumerate(words):
            if word == "ph" and i+1 < len(words) and words[i+1].replace('.', '', 1).isdigit():
                params['ph'] = float(words[i+1])
    
    # Similar parsing for other parameters
    param_keywords = {
        'organic carbon': 'oc',
        'nitrogen': 'n',
        'phosphorus': 'p2o5',
        'potassium': 'k20',
        'sulfur': 's',
        'copper': 'cu',
        'zinc': 'zn',
        'temperature': 'temperature',
        'humidity': 'humidity',
        'rainfall': 'rainfall'
    }
    
    for keyword, param in param_keywords.items():
        if keyword in text:
            words = text.split()
            for i, word in enumerate(words):
                if word == keyword and i+1 < len(words) and words[i+1].replace('.', '', 1).isdigit():
                    params[param] = float(words[i+1])
    
    return params

# Function to make prediction
def predict_crop(input_data, model_name):
    models, label_encoder, scaler = load_resources()
    
    # Scale the input data
    input_scaled = scaler.transform([input_data])
    
    # Reshape for LSTM input [samples, time steps, features]
    input_reshaped = input_scaled.reshape(1, 1, input_scaled.shape[1])
    
    # Make prediction
    model = models[model_name]
    prediction = model.predict(input_reshaped)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_crop = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get confidence score
    confidence = prediction[0][predicted_class] * 100
    
    return predicted_crop, confidence

# Main app
def main():
    st.title("🌱 AgriTech Crop Recommendation System")
    st.write("Get crop recommendations based on soil and climate conditions using deep learning models.")
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model_name = st.sidebar.radio(
        "Choose a Deep Learning Model",
        ["LSTM", "BiLSTM", "GRU"]
    )
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Text Input", "Voice Input"]
    )
    
    # Output method selection
    output_method = st.sidebar.multiselect(
        "Choose Output Method(s)",
        ["Text Output", "Voice Output"],
        default=["Text Output"]
    )
    
    # Initialize session state for input parameters
    if 'params' not in st.session_state:
        st.session_state.params = {
            'ph': 6.5,
            'oc': 0.5,
            'n': 50.0,
            'p2o5': 30.0,
            'k20': 200.0,
            's': 10.0,
            'cu': 0.5,
            'zn': 0.5,
            'temperature': 25.0,
            'humidity': 80.0,
            'rainfall': 200.0
        }
    
    # Handle voice input
    if input_method == "Voice Input":
        st.subheader("Voice Input")
        if st.button("Start Voice Input"):
            voice_text = voice_input()
            if voice_text:
                st.success(f"You said: {voice_text}")
                parsed_params = parse_voice_input(voice_text)
                
                if parsed_params:
                    for param, value in parsed_params.items():
                        st.session_state.params[param] = value
                    st.success("Parameters updated from voice input!")
                else:
                    st.warning("Could not extract parameters from voice. Please try again or use text input.")
    
    # Text input form
    st.subheader("Soil and Climate Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.params['ph'] = st.slider("pH Value", 0.0, 14.0, st.session_state.params['ph'], 0.1)
        st.session_state.params['oc'] = st.slider("Organic Carbon (OC)", 0.0, 2.0, st.session_state.params['oc'], 0.01)
        st.session_state.params['n'] = st.slider("Nitrogen (N)", 0.0, 200.0, st.session_state.params['n'], 1.0)
        st.session_state.params['p2o5'] = st.slider("Phosphorus (P2O5)", 0.0, 200.0, st.session_state.params['p2o5'], 1.0)
    
    with col2:
        st.session_state.params['k20'] = st.slider("Potassium (K20)", 0.0, 500.0, st.session_state.params['k20'], 1.0)
        st.session_state.params['s'] = st.slider("Sulfur (S)", 0.0, 100.0, st.session_state.params['s'], 0.1)
        st.session_state.params['cu'] = st.slider("Copper (CU)", 0.0, 10.0, st.session_state.params['cu'], 0.01)
        st.session_state.params['zn'] = st.slider("Zinc (ZN)", 0.0, 10.0, st.session_state.params['zn'], 0.01)
    
    with col3:
        st.session_state.params['temperature'] = st.slider("Temperature (°C)", 0.0, 50.0, st.session_state.params['temperature'], 0.1)
        st.session_state.params['humidity'] = st.slider("Humidity (%)", 0.0, 100.0, st.session_state.params['humidity'], 0.1)
        st.session_state.params['rainfall'] = st.slider("Rainfall (mm)", 0.0, 3000.0, st.session_state.params['rainfall'], 10.0)
    
    # Prediction button
    if st.button("Get Crop Recommendation"):
        with st.spinner(f"Analyzing with {model_name} model..."):
            # Prepare input data in the correct order
            input_data = [
                st.session_state.params['ph'],
                st.session_state.params['oc'],
                st.session_state.params['n'],
                st.session_state.params['p2o5'],
                st.session_state.params['k20'],
                st.session_state.params['s'],
                st.session_state.params['cu'],
                st.session_state.params['zn'],
                st.session_state.params['temperature'],
                st.session_state.params['humidity'],
                st.session_state.params['rainfall']
            ]
            
            # Make prediction
            predicted_crop, confidence = predict_crop(input_data, model_name)
            
            # Display results
            st.subheader("Recommendation Results")
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                st.success(f"Recommended Crop: **{predicted_crop}**")
                st.info(f"Confidence: {confidence:.2f}%")
                
                # Additional information about the crop
                st.subheader(f"About {predicted_crop}")
                st.write(f"Here's some information about {predicted_crop} cultivation:")
                
                # This would be replaced with actual crop information in a production system
                st.write(f"• {predicted_crop} is suitable for your soil and climate conditions.")
                st.write("• Follow proper planting and irrigation practices for best results.")
                st.write("• Consult with local agricultural experts for specific guidance.")
            
            with result_col2:
                # Placeholder for crop image
                st.image("https://via.placeholder.com/300x200?text=Crop+Image", caption=f"{predicted_crop} Image")
            
            # Voice output if selected
            if "Voice Output" in output_method:
                output_text = f"Based on the provided soil and climate parameters, I recommend growing {predicted_crop} with a confidence of {confidence:.0f} percent."
                st.markdown(text_to_speech(output_text), unsafe_allow_html=True)
    
    # Information section
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application uses deep learning models (LSTM, BiLSTM, and GRU) "
        "to recommend suitable crops based on soil parameters and climate conditions. "
        "The models were trained on agricultural data to provide accurate recommendations."
    )

if __name__ == "__main__":
    main()