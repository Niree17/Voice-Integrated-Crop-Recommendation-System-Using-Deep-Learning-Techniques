import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from tensorflow.keras.utils import to_categorical

# Load the dataset
print("Loading dataset...")
df = pd.read_excel('dataset.xlsx')

# Data preprocessing
print("Preprocessing data...")

# Handle missing values
df['Humidity'].fillna(df['Humidity'].mean(), inplace=True)

# Convert object columns to numeric where possible
for col in ['EC', 'FC', 'MN', 'BA']:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].mean(), inplace=True)
    except:
        print(f"Could not convert {col} to numeric")

# Select features for model training
features = ['PH', 'OC', 'N', 'P2O5', 'K20', 'S', 'CU', 'ZN', 'Temparature', 'Humidity', 'Rainfall']
X = df[features].values
y = df['CROP'].values

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
y_categorical = to_categorical(y_encoded)

# Save the label encoder for later use
joblib.dump(label_encoder, 'saved_models/label_encoder.joblib')

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, 'saved_models/scaler.joblib')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Reshape input for LSTM [samples, time steps, features]
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define model building functions
def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bilstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gru_model(input_shape, num_classes):
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train LSTM model
print("Training LSTM model...")
lstm_model = build_lstm_model((1, X_train.shape[1]), num_classes)
lstm_history = lstm_model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)
lstm_model.save('saved_models/LSTM.keras')
print(f"LSTM model accuracy: {lstm_model.evaluate(X_test_reshaped, y_test)[1]:.4f}")

# Train BiLSTM model
print("Training BiLSTM model...")
bilstm_model = build_bilstm_model((1, X_train.shape[1]), num_classes)
bilstm_history = bilstm_model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)
bilstm_model.save('saved_models/BiLSTM.keras')
print(f"BiLSTM model accuracy: {bilstm_model.evaluate(X_test_reshaped, y_test)[1]:.4f}")

# Train GRU model
print("Training GRU model...")
gru_model = build_gru_model((1, X_train.shape[1]), num_classes)
gru_history = gru_model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)
gru_model.save('saved_models/GRU.keras')
print(f"GRU model accuracy: {gru_model.evaluate(X_test_reshaped, y_test)[1]:.4f}")

print("All models trained and saved successfully!")