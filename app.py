import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

model = load_model("model_file.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

st.title("🧠 Facial Emotion Recognition App")
mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Live Webcam"])

if mode == "Image Upload":
    st.subheader("📷 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            resized = cv2.resize(roi_gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result)
            confidence = np.max(result)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, f"{labels_dict[label]} ({confidence:.2f})", (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        st.image(frame, caption="Detected Faces", channels="RGB")

elif mode == "Live Webcam":
    st.subheader("🎥 Live Emotion Detection from Webcam")

    class EmotionDetector(VideoTransformerBase):
        def __init__(self):
            self.model = load_model("model_file.h5")  # moved inside class
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                resized = cv2.resize(roi_gray, (48, 48))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                result = self.model.predict(reshaped)
                label = np.argmax(result)
                confidence = np.max(result)

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(img, f"{self.labels_dict[label]} ({confidence:.2f})", (x + 5, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            return img

    webrtc_streamer(key="emotion-stream", video_transformer_factory=EmotionDetector)
