from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Function to prepare input data from audio file
def prepare_input_data(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    input_data = np.expand_dims(np.expand_dims(mfcc, axis=-1), axis=0)
    return input_data

# Function to make prediction using the provided model and audio file
def make_prediction_from_drive(model_path, audio_file):
    # Load the trained model from Google Drive
    model = load_model(model_path)

    input_data = prepare_input_data(audio_file)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

model_path = 'emotion.h5'
audio_file_path = 'C:\\Users\\Wihanga Dilantha\\Desktop\\Emotion Detection\\audio\\one.wav'
result = make_prediction_from_drive(model_path, audio_file_path)

# Mapping numerical labels to emotion names
emotion_mapping = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'pleasant surprise',
    6: 'sad'
}
# Assuming your model prediction is stored in the variable 'prediction'
predicted_label = result[0]  # Assuming the prediction is a single value

# Map the numerical label to emotion name
predicted_emotion = emotion_mapping[predicted_label]

# Print the predicted emotion
print(f"{predicted_emotion}")
print(result)

