"""
export_artifacts.py
==================
JALANKAN INI DI COLAB, sebagai CELL BARU di notebook
HANDS_model_tuning_3fold, SETELAH cell training 5-fold CV selesai
(atau setelah kamu load ulang `le` dan `all_results` dari pickle).

Ini bukan file yang dijalankan lokal / di server Streamlit — cukup
copy-paste isi file ini ke satu cell Colab lalu jalankan. Nanti akan
menghasilkan 2 file di RESULTS_DIR:
    - label_encoder.pkl
    - model_2_best.weights.h5   (bobot fold dengan akurasi test tertinggi)

Setelah itu, download kedua file itu (klik kanan di file browser Colab
-> Download) dan taruh di folder `artifacts/` project Streamlit-mu.
"""

import os
import pickle
import shutil
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# 2. Definisikan Path (Sesuai dengan direktori training terakhir)
DATASET_ROOT = "/content/drive/MyDrive/Dataset/Dataset_Final"
PIPELINE_DIR = os.path.join(DATASET_ROOT, "pipeline_v2_hands_spatiotemporal_training_3kali")
RESULTS_DIR  = os.path.join(PIPELINE_DIR, "results_subj_hands_focal_spatiotemporal_training_3kali")
DF_FINAL_PATH = os.path.join(PIPELINE_DIR, "df_fold_subj_hands_spatiotemporal_training_3kali.csv")
RESULTS_PKL_PATH = os.path.join(RESULTS_DIR, "all_results_hands_focal_spatiotemporal_training_3kali.pkl")

VID = "Model_4"
TARGET_SEED = 42

# 3. Re-create & Simpan Label Encoder
print("⏳ Membangun ulang Label Encoder dari dataset...")
df_final = pd.read_csv(DF_FINAL_PATH)
df_pool = df_final[(df_final["status"] == "ok") & (df_final["angle"].isin(["depan", "samping"]))].copy()

le = LabelEncoder()
le.fit(df_pool["gloss"].values)

encoder_out_path = os.path.join(RESULTS_DIR, "label_encoder.pkl")
with open(encoder_out_path, "wb") as f:
    pickle.dump(le, f)
print(f"✅ Label encoder disimpan -> {encoder_out_path}")

# 4. Load all_results dari Pickle untuk mencari iterasi terbaik
print("⏳ Membaca riwayat training...")
with open(RESULTS_PKL_PATH, "rb") as f:
    all_results = pickle.load(f)

fold_scores = [s for s in all_results[VID]["fold_scores"] if s["seed"] == TARGET_SEED]
best_fold = max(fold_scores, key=lambda s: s["accuracy_17"])
best_iterasi = best_fold["iterasi"]

# 5. Salin file weights terbaik dengan nama baku untuk deploy
src_weights = os.path.join(RESULTS_DIR, f"{VID}_iter{best_iterasi}_seed{TARGET_SEED}_hands_spatiotemporal.weights.h5")
dst_weights = os.path.join(RESULTS_DIR, "model_4_seed42_best.weights.h5")

if os.path.exists(src_weights):
    shutil.copyfile(src_weights, dst_weights)
    print(f"✅ Bobot terbaik (Iterasi {best_iterasi}, Seed {TARGET_SEED}) dengan akurasi uji {best_fold['accuracy_17']:.4f} disimpan -> {dst_weights}")
    print(f"\n📥 Silakan buka Google Drive Anda di folder '{RESULTS_DIR}' dan download dua file ini untuk Streamlit:")
    print(f"   1. label_encoder.pkl")
    print(f"   2. model_4_seed42_best.weights.h5")
else:
    print(f"⚠️ File tidak ditemukan: {src_weights}")