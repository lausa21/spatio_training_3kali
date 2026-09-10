import os
import pickle
import collections
import numpy as np
import pandas as pd
import streamlit as st
import av
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

from model_utils import (
    build_transformer_model, 
    extract_frame_features, 
    trim_idle_frames, 
    normalize_sequence
)

ARTIFACTS_DIR = "artifacts"
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_4_seed42_best.weights.h5")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

st.set_page_config(page_title="Deteksi Bahasa Isyarat", page_icon="🤟", layout="centered")

# 1. Pemuatan Artifacts
@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    if not os.path.exists(ENCODER_PATH) or not os.path.exists(WEIGHTS_PATH):
        st.error("File artifacts tidak lengkap.")
        st.stop()

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    model = build_transformer_model(num_classes=num_classes)
    model.load_weights(WEIGHTS_PATH)
    
    # Cache MediaPipe agar tidak membebani memori setiap frame
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return model, le, hands

model, le, hands = load_artifacts()

# 2. Inisialisasi Buffer Global (Spesifik per Sesi User)
if "frame_buffer" not in st.session_state:
    st.session_state.frame_buffer = collections.deque(maxlen=60)

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 3. Callback Kamera Real-Time
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    
    results = hands.process(img_rgb)
    features = extract_frame_features(results)
    
    # Terus masukkan fitur ke antrean memori
    st.session_state.frame_buffer.append(features)
    
    # Indikator visual di pojok kiri atas kamera
    buffer_len = len(st.session_state.frame_buffer)
    cv2.putText(img, f"Buffer: {buffer_len}/60 frames", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Antarmuka Pengguna
st.title("🤟 Live Deteksi Bahasa Isyarat Medis")
st.caption("Model: Transformer | Seed 42 | Input: 84-D | 30 frame")

st.info("1. Klik **Start** untuk menyalakan kamera.\n2. Lakukan gerakan isyarat (maksimal 2 detik).\n3. Tekan tombol **🚀 Prediksi Gerakan** di bawah kamera.")

webrtc_streamer(
    key="skripsi-live-kamera",
    rtc_configuration=RTC_CONFIG,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)

# 5. Logika Tombol Prediksi & Restart Buffer
if st.button("🚀 Prediksi Gerakan", use_container_width=True):
    # Kunci isi buffer saat tombol ditekan
    current_buffer = list(st.session_state.frame_buffer)
    
    # Langsung kosongkan (restart) buffer agar siap merekam isyarat berikutnya
    st.session_state.frame_buffer.clear()
    
    if len(current_buffer) < 15:
        st.warning("Gerakan terlalu singkat atau kamera baru saja dinyalakan. Silakan ulangi peragaan.")
    else:
        with st.spinner("Mengevaluasi sekuens isyarat..."):
            sequence_raw = np.array(current_buffer, dtype=np.float32)
            
            # Potong frame statis
            trimmed_seq, _ = trim_idle_frames(sequence_raw, threshold_ratio=0.15)
            
            if trimmed_seq.shape[0] < 5:
                st.error("Tidak ada pergerakan tangan yang terdeteksi. Pastikan tangan masuk ke dalam bingkai kamera.")
            else:
                # Normalisasi dan Prediksi
                norm_seq = normalize_sequence(trimmed_seq)
                input_data = np.expand_dims(norm_seq, axis=0)
                
                probs = model.predict(input_data, verbose=0)[0]
                top_idx = np.argsort(probs)[::-1][:5]
                labels = le.inverse_transform(top_idx)
                scores = probs[top_idx]
                results = list(zip(labels, scores))
                
                # Render Hasil (Top 1)
                st.success(f"Berhasil diproses! Kamera membaca {len(current_buffer)} frame.")
                top_label, top_score = results[0]
                st.metric("Prediksi Teratas", top_label, f"{top_score * 100:.1f}% keyakinan")
                
                # Render Hasil (Top 1-5 dengan Persentase)
                st.subheader("Top-5 Prediksi")
                df = pd.DataFrame(results, columns=["Kosakata", "Probabilitas"])
                df["Probabilitas"] = (df["Probabilitas"] * 100).round(2)
                
                st.bar_chart(df.set_index("Kosakata"))
                st.dataframe(df, hide_index=True, use_container_width=True)
