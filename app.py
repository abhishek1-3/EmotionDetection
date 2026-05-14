from pathlib import Path
from threading import Lock

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from tensorflow.keras.models import load_model


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_file.h5"
LOCAL_CASCADE_PATH = APP_DIR / "haarcascade_frontalface_default.xml"
LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}
LABEL_COLORS = {
    "Angry": (55, 65, 245),
    "Disgust": (35, 150, 95),
    "Fear": (220, 120, 35),
    "Happy": (45, 190, 120),
    "Neutral": (125, 125, 125),
    "Sad": (220, 105, 70),
    "Surprise": (205, 80, 210),
}
PREDICT_LOCK = Lock()


st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon=":)",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        [data-testid="stSidebar"] {
            background: #f7f8fb;
        }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color: #111827 !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label,
        [data-testid="stSidebar"] [data-testid="stSlider"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #111827 !important;
        }

        .app-header {
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
        }

        .app-header h1 {
            color: #111827;
            font-size: 2.25rem;
            letter-spacing: 0;
            line-height: 1.1;
            margin: 0;
        }

        .app-header p {
            color: #4b5563;
            font-size: 1rem;
            margin: .45rem 0 0;
        }

        .status-panel {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            color: #111827;
            padding: .85rem 1rem;
        }

        .status-panel strong,
        .status-panel h3,
        .status-panel div {
            color: #111827;
        }

        .emotion-pill {
            border-radius: 999px;
            color: #ffffff !important;
            display: inline-block;
            font-size: .88rem;
            font-weight: 700;
            margin: .2rem .35rem .2rem 0;
            padding: .35rem .7rem;
        }

        .small-muted {
            color: #6b7280;
            font-size: .9rem;
        }

        .face-card {
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading emotion model...")
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    # compile=False avoids legacy H5 compile-state deserialization issues.
    return load_model(MODEL_PATH, compile=False)


@st.cache_resource(show_spinner="Loading face detector...")
def get_face_detector():
    cascade_path = LOCAL_CASCADE_PATH
    if not cascade_path.exists():
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    return detector


def predict_emotion(model, face_gray):
    resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    batch = np.expand_dims(normalized, axis=(0, -1))

    with PREDICT_LOCK:
        scores = model.predict(batch, verbose=0)[0]

    label_index = int(np.argmax(scores))
    label = LABELS[label_index]
    confidence = float(scores[label_index])
    return label, confidence


def draw_label(frame, x, y, w, h, label, confidence):
    color = LABEL_COLORS.get(label, (55, 160, 240))
    text = f"{label} {confidence * 100:.0f}%"
    text_y = max(y - 10, 24)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, text_y - 24), (x + max(w, 150), text_y + 6), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 8, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def detect_emotions(frame, model, detector, scale_factor, min_neighbors):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
    )

    detections = []
    frame_height, frame_width = frame.shape[:2]
    for x, y, w, h in faces:
        pad_x = max(int(w * 0.12), 8)
        pad_y = max(int(h * 0.12), 8)
        crop_x1 = max(int(x) - pad_x, 0)
        crop_y1 = max(int(y) - pad_y, 0)
        crop_x2 = min(int(x + w) + pad_x, frame_width)
        crop_y2 = min(int(y + h) + pad_y, frame_height)
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        face_gray = gray[y : y + h, x : x + w]
        label, confidence = predict_emotion(model, face_gray)
        detections.append(
            {
                "label": label,
                "confidence": confidence,
                "box": (int(x), int(y), int(w), int(h)),
                "face": face_crop,
            }
        )
        draw_label(frame, x, y, w, h, label, confidence)

    return frame, detections


def render_detection_summary(detections):
    if not detections:
        st.info("No faces detected. Try a brighter image or face the camera directly.")
        return

    st.success(f"Detected {len(detections)} face{'s' if len(detections) != 1 else ''}.")
    cols = st.columns(min(len(detections), 3))
    for index, detection in enumerate(detections):
        label = detection["label"]
        confidence = detection["confidence"]
        color = LABEL_COLORS.get(label, (80, 80, 80))
        hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        with cols[index % len(cols)]:
            st.image(
                detection["face"],
                caption=f"Face {index + 1}",
                use_container_width=True,
            )
            st.markdown(
                f"""
                <div class="status-panel face-card">
                    <div class="emotion-pill" style="background:{hex_color}; color:white;">
                        {label}
                    </div>
                    <div class="small-muted">Confidence</div>
                    <h3 style="margin:.1rem 0 0;">{confidence * 100:.1f}%</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )


class EmotionDetector(VideoProcessorBase):
    def __init__(self, model, detector, scale_factor, min_neighbors):
        self.model = model
        self.detector = detector
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        processed_rgb, _ = detect_emotions(
            img_rgb,
            self.model,
            self.detector,
            self.scale_factor,
            self.min_neighbors,
        )
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(processed_bgr, format="bgr24")


st.markdown(
    """
    <div class="app-header">
        <h1>Facial Emotion Recognition</h1>
        <p>Upload a photo or use your webcam to detect facial expressions in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Image Upload", "Live Webcam"], label_visibility="collapsed")
    st.divider()
    scale_factor = st.slider("Face scan scale", 1.05, 1.50, 1.25, 0.05)
    min_neighbors = st.slider("Detection strictness", 3, 10, 5, 1)
    st.caption("Lower strictness finds more faces. Higher strictness reduces false detections.")

try:
    model = get_model()
    face_detector = get_face_detector()
except Exception as exc:
    st.error("The app could not start because a required model or face detector failed to load.")
    st.exception(exc)
    st.stop()

left_col, right_col = st.columns([1.05, 1], gap="large")

if mode == "Image Upload":
    with left_col:
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a face image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is None:
            st.markdown(
                '<div class="status-panel">Add a JPG or PNG image to start detection.</div>',
                unsafe_allow_html=True,
            )
        else:
            image = Image.open(uploaded_file).convert("RGB")
            frame = np.array(image)
            processed_frame, detections = detect_emotions(
                frame.copy(),
                model,
                face_detector,
                scale_factor,
                min_neighbors,
            )
            st.image(processed_frame, caption="Detection result", use_container_width=True)

    with right_col:
        st.subheader("Result")
        if uploaded_file is None:
            st.markdown(
                """
                <div class="status-panel">
                    <strong>Ready when you are.</strong>
                    <div class="small-muted">Results will appear here after you upload an image.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_detection_summary(detections)

else:
    with left_col:
        st.subheader("Live Webcam")
        st.caption("Allow camera access when your browser asks. Disable VPNs if WebRTC cannot connect.")

        processor_factory = lambda: EmotionDetector(
            model,
            face_detector,
            scale_factor,
            min_neighbors,
        )
        webrtc_streamer(
            key="emotion-stream",
            video_processor_factory=processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with right_col:
        st.subheader("Live Status")
        st.markdown(
            """
            <div class="status-panel">
                <strong>Webcam mode is active.</strong>
                <div class="small-muted">
                    The model is loaded once and reused for every frame to avoid Keras reload errors.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Emotion classes")
        st.write(", ".join(LABELS.values()))
