"""
app.py
==================
Aplikasi Streamlit untuk deploy model Transformer (Tanpa Augmentasi)
hasil training HANDS_model_tuning_3fold.

Cara pakai:
    streamlit run app.py

Sebelum dijalankan, pastikan 2 file berikut ada di folder `artifacts/`:
    - artifacts/model_2_best.weights.h5
    - artifacts/label_encoder.pkl
(lihat export_artifacts.py untuk cara membuatnya dari Colab)
"""

# checkpoint

import os
import cv2
import pickle
import tempfile
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
from collections import deque

# Pastikan fungsi-fungsi ini diimpor dari model_utils.py milikmu
from model_utils import (
    build_transformer_model,
    video_to_model_input,
    extract_frame_features,
    trim_idle_frames,      
    normalize_sequence,     
    TARGET_SEQ_LEN,
    FEATURE_DIM,
    mp_hands,
    MEDIAPIPE_CONFIG
)

ARTIFACTS_DIR = "artifacts"
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_4_seed42_best.weights.h5")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

st.set_page_config(page_title="Deteksi Bahasa Isyarat", page_icon="🤟", layout="centered")

@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    if not os.path.exists(ENCODER_PATH) or not os.path.exists(WEIGHTS_PATH):
        st.error(f"File artifacts tidak lengkap. Pastikan file ada di {ARTIFACTS_DIR}/")
        st.stop()

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    model = build_transformer_model(num_classes=num_classes)
    model.load_weights(WEIGHTS_PATH)
    return model, le

def predict_video(model, le, filepath, top_k=5):
    model_input, raw_seq = video_to_model_input(filepath)
    if model_input is None or raw_seq.shape[0] == 0:
        return None, None

    probs = model.predict(model_input, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    labels = le.inverse_transform(top_idx)
    scores = probs[top_idx]
    return list(zip(labels, scores)), raw_seq.shape[0]

def run_realtime_inference(model, le):
    st.warning("Tekan tombol **'Stop' di pojok kanan atas** untuk mematikan kamera. Matikan tangan dari frame untuk melihat hasil akhir.")
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # PERUBAHAN: Buffer sekarang berupa list kosong TANPA batas maksimal
    sequence_buffer = [] 
    empty_frames = 0
    
    state = "WAITING"
    locked_predictions = []
    lock_timer = 0

    with mp_hands.Hands(**MEDIAPIPE_CONFIG) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                st.error("Gagal membaca kamera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                empty_frames = 0
                if state == "RESULT_LOCKED":
                    state = "RECORDING"
                    sequence_buffer.clear()
                    
                state = "RECORDING"
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                features = extract_frame_features(results)
                sequence_buffer.append(features) # Rekam semua frame selama tangan ada di layar
            else:
                empty_frames += 1

            if empty_frames > 150:
                st.info("Kamera otomatis dimatikan karena idle.")
                break

            # Trigger Penguncian Hasil (Tangan turun)
            if empty_frames > 5 and state == "RECORDING":
                # Hanya proses jika user merekam minimal 15 frame (~0.5 detik) untuk menghindari noise/kedipan
                if len(sequence_buffer) >= 15:
                    # 1. Jadikan matriks mentah utuh (berapapun panjangnya)
                    raw_seq = np.array(sequence_buffer, dtype=np.float32)
                    
                    # 2. Potong frame diam (idle) dari awal/akhir gerakan menggunakan kodemu
                    trimmed_seq, _ = trim_idle_frames(raw_seq)
                    
                    # 3. Normalisasi temporal (paksa jadi 30 frame) & spasial
                    norm_seq = normalize_sequence(trimmed_seq)
                    
                    # 4. Prediksi
                    probs = model.predict(np.expand_dims(norm_seq, axis=0), verbose=0)[0]
                    
                    top_idx = np.argsort(probs)[::-1][:3]
                    locked_predictions = list(zip(le.inverse_transform(top_idx), probs[top_idx]))
                    
                    state = "RESULT_LOCKED"
                    lock_timer = 90
                else:
                    state = "WAITING"
                
                sequence_buffer.clear()

            if state == "RESULT_LOCKED":
                lock_timer -= 1
                if lock_timer <= 0:
                    state = "WAITING"
                    locked_predictions = []

            # ---------------- UI OVERLAY ----------------
            cv2.rectangle(frame, (0, 0), (640, 130), (35, 35, 35), -1)
            
            if state == "WAITING":
                cv2.putText(frame, "Menunggu Gerakan...", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            elif state == "RECORDING":
                # Indikator sekarang menampilkan jumlah frame aktual, bukan persentase
                frame_count = len(sequence_buffer)
                cv2.putText(frame, f"Merekam... ({frame_count} frames)", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif state == "RESULT_LOCKED" and locked_predictions:
                for i, (lbl, score) in enumerate(locked_predictions):
                    color = (0, 255, 0) if i == 0 else (200, 200, 200)
                    font_scale = 0.9 if i == 0 else 0.7
                    thickness = 2 if i == 0 else 1
                    text = f"{i+1}. {lbl} ({score*100:.1f}%)"
                    cv2.putText(frame, text, (15, 35 + (i * 35)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            frame_rgb_render = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb_render, channels="RGB", use_container_width=True)

    cap.release()

# SEMACAM REPORT
# def run_realtime_inference(model, le):
#     st.warning("Tekan tombol **'Stop' di pojok kanan atas** untuk mematikan kamera. Matikan tangan dari frame untuk melihat hasil akhir.")
#     frame_placeholder = st.empty()
    
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     sequence_buffer = deque(maxlen=TARGET_SEQ_LEN)
#     empty_frames = 0
    
#     # Variabel State Machine
#     state = "WAITING"
#     locked_predictions = []
#     lock_timer = 0

#     with mp_hands.Hands(**MEDIAPIPE_CONFIG) as hands:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret: 
#                 st.error("Gagal membaca kamera.")
#                 break

#             frame = cv2.flip(frame, 1)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb_frame)

#             if results.multi_hand_landmarks:
#                 empty_frames = 0
#                 # Jika tangan kembali muncul, paksa kembali ke fase merekam
#                 if state == "RESULT_LOCKED":
#                     state = "RECORDING"
#                     sequence_buffer.clear()
                    
#                 state = "RECORDING"
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 features = extract_frame_features(results)
#                 sequence_buffer.append(features)
#             else:
#                 empty_frames += 1

#             # Auto-Stop jika ditinggal > 5 detik
#             if empty_frames > 150:
#                 st.info("Kamera otomatis dimatikan karena idle.")
#                 break

#             # Trigger Penguncian Hasil (Tangan turun setelah gerakan penuh)
#             if empty_frames > 5 and state == "RECORDING":
#                 if len(sequence_buffer) == TARGET_SEQ_LEN:
#                     # Lakukan Prediksi
#                     seq_array = np.array(sequence_buffer, dtype=np.float32)
#                     norm_seq = spatial_normalize_hands(seq_array)
#                     probs = model.predict(np.expand_dims(norm_seq, axis=0), verbose=0)[0]
                    
#                     top_idx = np.argsort(probs)[::-1][:3]
#                     locked_predictions = list(zip(le.inverse_transform(top_idx), probs[top_idx]))
                    
#                     # Ubah state & atur timer (90 frame ~ 3 detik)
#                     state = "RESULT_LOCKED"
#                     lock_timer = 90
#                 else:
#                     # Gerakan terlalu pendek, batalkan
#                     state = "WAITING"
                
#                 sequence_buffer.clear()

#             # Timer pengurangan untuk menghilangkan hasil dari layar
#             if state == "RESULT_LOCKED":
#                 lock_timer -= 1
#                 if lock_timer <= 0:
#                     state = "WAITING"
#                     locked_predictions = []

#             # ---------------- UI OVERLAY ----------------
#             cv2.rectangle(frame, (0, 0), (640, 130), (35, 35, 35), -1)
            
#             if state == "WAITING":
#                 cv2.putText(frame, "Menunggu Gerakan...", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
#             elif state == "RECORDING":
#                 progress = int((len(sequence_buffer) / TARGET_SEQ_LEN) * 100)
#                 cv2.putText(frame, f"Merekam... {progress}%", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#             elif state == "RESULT_LOCKED" and locked_predictions:
#                 for i, (lbl, score) in enumerate(locked_predictions):
#                     color = (0, 255, 0) if i == 0 else (200, 200, 200)
#                     font_scale = 0.9 if i == 0 else 0.7
#                     thickness = 2 if i == 0 else 1
#                     text = f"{i+1}. {lbl} ({score*100:.1f}%)"
#                     cv2.putText(frame, text, (15, 35 + (i * 35)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

#             frame_rgb_render = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb_render, channels="RGB", use_container_width=True)

#     cap.release()

# REAL TIME
# def run_realtime_inference(model, le):
#     st.warning("Tekan tombol **'Stop' di pojok kanan atas** untuk mematikan kamera. Kamera juga akan mati otomatis jika tidak ada gerakan selama 5 detik.")
#     frame_placeholder = st.empty()
    
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     sequence_buffer = deque(maxlen=TARGET_SEQ_LEN)
#     empty_frames = 0
#     top_predictions = []  # Menyimpan list top-3 prediksi

#     with mp_hands.Hands(**MEDIAPIPE_CONFIG) as hands:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret: 
#                 st.error("Gagal membaca kamera.")
#                 break

#             # Mirror frame
#             frame = cv2.flip(frame, 1)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb_frame)

#             # Algoritma Debounce & Landmark Drawing
#             if results.multi_hand_landmarks:
#                 empty_frames = 0
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             else:
#                 empty_frames += 1

#             # 1. Fitur Auto-Stop: Matikan kamera jika idle ~5 detik (150 frames)
#             if empty_frames > 150:
#                 st.info("Kamera otomatis dimatikan karena tidak mendeteksi tangan selama 5 detik.")
#                 break

#             # Bersihkan buffer jika tangan sempat turun sejenak
#             if empty_frames > 10:
#                 sequence_buffer.clear()
#                 top_predictions = []

#             features = extract_frame_features(results)
#             sequence_buffer.append(features)

#             # 2. Fitur Continuous Listing: Prediksi selalu jalan jika 30 frame terpenuhi
#             if len(sequence_buffer) == TARGET_SEQ_LEN and empty_frames == 0:
#                 seq_array = np.array(sequence_buffer, dtype=np.float32)
#                 norm_seq = spatial_normalize_hands(seq_array)
                
#                 probs = model.predict(np.expand_dims(norm_seq, axis=0), verbose=0)[0]
                
#                 # Ambil 3 indeks dengan nilai probabilitas tertinggi
#                 top_k = 3
#                 top_idx = np.argsort(probs)[::-1][:top_k]
#                 top_labels = le.inverse_transform(top_idx)
#                 top_scores = probs[top_idx]
                
#                 top_predictions = list(zip(top_labels, top_scores))

#             # Overlay UI pada Frame (Diperluas untuk 3 baris teks)
#             cv2.rectangle(frame, (0, 0), (640, 130), (35, 35, 35), -1)
            
#             if top_predictions:
#                 for i, (lbl, score) in enumerate(top_predictions):
#                     # Highlight hijau terang untuk Prediksi #1, abu-abu untuk sisanya
#                     color = (0, 255, 0) if i == 0 else (200, 200, 200)
#                     font_scale = 0.9 if i == 0 else 0.7
#                     thickness = 2 if i == 0 else 1
                    
#                     text = f"{i+1}. {lbl} ({score*100:.1f}%)"
#                     cv2.putText(frame, text, (15, 35 + (i * 35)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
#             else:
#                 cv2.putText(frame, "Menunggu Gerakan...", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

#             frame_rgb_render = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb_render, channels="RGB", use_container_width=True)

#     cap.release()

def main():
    st.title("🤟 Deteksi Kosakata Bahasa Isyarat")
    model, le = load_artifacts()

    mode = st.sidebar.radio("Pilih Mode Operasi:", ["Kamera Real-Time", "Upload Video"])
    
    st.sidebar.info(f"Jumlah kosakata terdaftar: **{len(le.classes_)}**")
    with st.sidebar.expander("Lihat daftar kosakata"):
        st.write(", ".join(sorted(le.classes_)))

    if mode == "Kamera Real-Time":
        st.subheader("🔴 Deteksi Real-Time (Webcam)")
        if st.button("Mulai Kamera", use_container_width=True):
            run_realtime_inference(model, le)
            
    else:
        st.subheader("📁 Prediksi via Video")
        uploaded_file = st.file_uploader("Upload video isyarat tangan (.mp4)", type=["mp4", "mov", "avi"])

        if uploaded_file is not None:
            st.video(uploaded_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Mengekstrak landmark tangan & memprediksi..."):
                    results, n_frames = predict_video(model, le, tmp_path, top_k=5)
            finally:
                os.unlink(tmp_path)

            if results is None:
                st.warning("Tidak ada tangan yang terdeteksi di video ini.")
                return

            top_label, top_score = results[0]
            st.metric("Prediksi Teratas", top_label, f"{top_score * 100:.1f}% keyakinan")

            st.subheader("Top-5 Prediksi")
            df = pd.DataFrame(results, columns=["Kosakata", "Probabilitas"])
            df["Probabilitas"] = (df["Probabilitas"] * 100).round(2)
            st.dataframe(df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()