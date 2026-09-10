import os
import pickle
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import build_transformer_model, video_to_model_input, TARGET_SEQ_LEN, FEATURE_DIM

ARTIFACTS_DIR = "artifacts"
WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "model_4_seed42_best.weights.h5")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

st.set_page_config(page_title="Deteksi Bahasa Isyarat (Hands)", page_icon="🤟", layout="centered")

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
    return model, le

def predict(model, le, filepath, top_k=5):
    model_input, raw_seq = video_to_model_input(filepath)
    if model_input is None or raw_seq.shape[0] == 0:
        return None, None

    probs = model.predict(model_input, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    labels = le.inverse_transform(top_idx)
    scores = probs[top_idx]
    return list(zip(labels, scores)), raw_seq.shape[0]

def main():
    st.title("🤟 Deteksi Kosakata Bahasa Isyarat")
    st.caption("Model: Transformer (Dengan Augmentasi) | Seed 42 | Input: 84-D | 30 frame")

    model, le = load_artifacts()

    st.info(f"Model siap. Jumlah kosakata terdaftar: **{len(le.classes_)}**")
    with st.expander("Lihat daftar kosakata yang dikenali model"):
        st.write(", ".join(sorted(le.classes_)))

    uploaded_file = st.file_uploader(
        "Upload video isyarat tangan (.mp4)",
        type=["mp4", "mov", "avi"],
        help="Idealnya video berisi satu gerakan/kosakata, tangan terlihat jelas."
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Mengekstrak landmark tangan & memprediksi..."):
                results, n_frames = predict(model, le, tmp_path, top_k=5)
        finally:
            os.unlink(tmp_path)

        if results is None:
            st.warning("Tidak ada tangan yang terdeteksi di video ini.")
            return

        st.success(f"Video diproses ({n_frames} frame terbaca).")
        top_label, top_score = results[0]
        st.metric("Prediksi Teratas", top_label, f"{top_score * 100:.1f}% keyakinan")

        st.subheader("Top-5 Prediksi")
        df = pd.DataFrame(results, columns=["Kosakata", "Probabilitas"])
        df["Probabilitas"] = (df["Probabilitas"] * 100).round(2)
        st.bar_chart(df.set_index("Kosakata"))
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.write("👆 Upload video untuk mulai prediksi.")

if __name__ == "__main__":
    main()
