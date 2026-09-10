# Deploy Model Transformer (Hands, Tanpa Aug) ke Streamlit

Model yang dipakai: **Model_2 — "Transformer (Tanpa Aug)"** dari notebook
`HANDS_model_tuning_3fold`. Input 84-D landmark tangan (MediaPipe Hands,
21 titik x 2 koordinat x 2 tangan), 30 frame per sekuens, output 17 kelas.

## Struktur folder

```
streamlit_deploy/
├── app.py                 # aplikasi utama Streamlit
├── model_utils.py         # preprocessing + arsitektur model (jangan diubah)
├── export_artifacts.py    # dijalankan DI COLAB, bukan di sini
├── requirements.txt
├── packages.txt            # dependency sistem untuk Streamlit Cloud
├── artifacts/               # kamu buat manual, isi 2 file di bawah
│   ├── model_2_best.weights.h5
│   └── label_encoder.pkl
└── README.md
```

## Langkah 1 — Export aset dari Colab

Kamu **hanya menyimpan bobot (`.weights.h5`)**, belum ada `label_encoder.pkl`
dan belum ada bobot dengan nama baku yang dipakai app. Jadi:

1. Buka lagi notebook `HANDS_model_tuning_3fold` di Colab.
2. Jalankan sampai cell training 5-fold CV selesai (atau load ulang
   `all_results` dan `le` dari pickle `RESULTS_PKL_PATH` kalau sudah pernah training).
3. Buat cell baru, copy-paste isi `export_artifacts.py`, jalankan.
4. Script itu akan otomatis:
   - Menyimpan `label_encoder.pkl` (berisi daftar 17 kosakata).
   - Memilih **fold dengan akurasi test tertinggi** untuk Model_2, lalu
     menyalinnya sebagai `model_2_best.weights.h5`.
5. Download kedua file itu dari Colab (klik kanan di file browser → Download),
   lalu taruh di folder `artifacts/` di project ini.

> Kenapa harus pilih 1 fold? Karena training kamu pakai 5-fold CV — ada 5
> file bobot berbeda (`Model_2_iter1_hands.weights.h5` ... `iter5`). Untuk
> deploy, dipilih otomatis yang akurasi test-nya paling tinggi. Kalau kamu
> mau fold tertentu (misal yang paling "general", bukan cuma tertinggi),
> tinggal ubah baris `best_fold = max(...)` di `export_artifacts.py`.

## Langkah 2 — Install & jalankan lokal

```bash
cmd
py -m venv venv
venv\Scripts\activate atau venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`, upload video `.mp4` berisi satu gerakan
tangan, lihat hasil prediksi top-5.

## Langkah 3 — Deploy ke Streamlit Community Cloud (opsional)

1. Push folder ini ke repo GitHub (sertakan `artifacts/*.h5` dan `*.pkl` —
   kalau ukurannya besar, pertimbangkan Git LFS).
2. Buka [share.streamlit.io](https://share.streamlit.io), connect ke repo,
   pilih `app.py` sebagai entry point.
3. `packages.txt` akan otomatis dipakai Streamlit Cloud untuk install
   dependency sistem (`libgl1`, dll) yang dibutuhkan OpenCV.

## Catatan penting soal preprocessing di app.py

Notebook training pakai `split_and_trim_video()` yang memecah 1 video jadi
2 sekuens (sudut depan & samping) karena data trainingmu memang direkam
begitu. Untuk inferensi user upload video baru (1 gerakan saja), app ini
pakai versi yang lebih sederhana: `trim_idle_frames()` → `normalize_sequence()`
langsung pada seluruh video, TANPA split dua sudut. Ini ada di
`video_to_model_input()` dalam `model_utils.py`. Kalau ternyata video
inferensi kamu juga berisi 2 gerakan (depan+samping) dalam 1 file, kasih
tahu aku, nanti fungsi ini aku sesuaikan supaya juga split.

## Troubleshooting cepat

| Masalah | Kemungkinan penyebab |
|---|---|
| Error load_weights (shape mismatch) | Pastikan `BEST_HP_TRANSFORMER_NOAUG` di `model_utils.py` sama persis dengan hasil tuning di notebook kamu |
| "Tidak ada tangan terdeteksi" terus | Video terlalu gelap/tangan di luar frame; coba turunkan `min_detection_confidence` di `MEDIAPIPE_CONFIG` |
| Streamlit Cloud gagal install opencv | Pastikan `packages.txt` ikut ter-push ke repo |
| Prediksi ngawur / confidence rendah semua | Cek apakah `label_encoder.pkl` yang dipakai benar-benar cocok dengan bobot model (harus dari run training yang sama) |
