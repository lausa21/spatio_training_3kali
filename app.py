import streamlit as st
import cv2
import numpy as np
import collections
import pickle
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Impor fungsi preprocessing dan arsitektur dari model_utils.py
from model_utils import (
    build_transformer_model_v2,  # Ubah menjadi v2
    extract_frame_features,
    trim_idle_frames,
    normalize_sequence,
    TARGET_SEQ_LEN, 
    FEATURE_DIM    
    # NUM_CLASSES dihapus dari daftar impor
)

# Deklarasikan NUM_CLASSES secara manual di sini
NUM_CLASSES = 17 

# 1. Pemuatan Artifacts Berbasis Cache
@st.cache_resource
def load_artifacts():
    with open("artifacts/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    # Gunakan pemanggil fungsi v2
    model = build_transformer_model_v2(
        seq_len=TARGET_SEQ_LEN, 
        feature_dim=FEATURE_DIM, 
        num_classes=NUM_CLASSES
    )
    model.load_weights("artifacts/model_2_best.weights.h5")
    return le, model

le, model = load_artifacts()

# 2. Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# 3. Buffer Dinamis (60 Frame / 2 Detik)
frame_buffer = collections.deque(maxlen=60)
prediction_text = "Menunggu gerakan isyarat..."

# 4. Konfigurasi STUN Server
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# 5. Callback Inferensi Real-Time
def video_frame_callback(frame):
    global prediction_text
    img = frame.to_ndarray(format="bgr24")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = hands.process(img_rgb)
    
    features = extract_frame_features(results)
    frame_buffer.append(features)
    
    # Picu evaluasi jika buffer sudah menampung 60 frame
    if len(frame_buffer) == 60:
        sequence_raw = np.array(frame_buffer, dtype=np.float32)
        
        # Potong frame statis di awal dan akhir peragaan
        sequence_trimmed, _ = trim_idle_frames(sequence_raw, threshold_ratio=0.15)
        
        # Validasi sisa frame (minimal 15 frame bergerak untuk diprediksi)
        if sequence_trimmed.shape[0] >= 15:
            # Susutkan/Tarik menjadi 30 frame secara proporsional
            norm_seq = normalize_sequence(sequence_trimmed)
            input_data = np.expand_dims(norm_seq, axis=0)
            
            y_pred_proba = model.predict(input_data, verbose=0)
            y_pred_idx = np.argmax(y_pred_proba, axis=1)
            confidence = np.max(y_pred_proba)
            
            if confidence > 0.60:
                label_teks = le.inverse_transform(y_pred_idx)[0]
                prediction_text = f"Deteksi: {label_teks} ({confidence:.2f})"
            
            # Bersihkan buffer setelah prediksi berhasil
            frame_buffer.clear()
        else:
            # Jika tidak ada gerakan, buang separuh isi buffer terlama
            # agar antrean terus bergeser mencari pergerakan baru
            for _ in range(30):
                frame_buffer.popleft()
                
    # UI Teks di Frame Kamera
    cv2.rectangle(img, (0, 0), (640, 60), (14, 52, 112), -1)
    cv2.putText(img, prediction_text, (15, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 6. Antarmuka Web
st.set_page_config(page_title="Deteksi Isyarat Medis", layout="wide")
st.title("Sistem Deteksi Bahasa Isyarat Medis Berbasis Transformer")

webrtc_streamer(
    key="skripsi-deteksi",
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)
