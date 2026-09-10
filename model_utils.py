import math
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

TARGET_SEQ_LEN = 30
FEATURE_DIM    = 84
RHAND_SLICE    = slice(0, 42)
LHAND_SLICE    = slice(42, 84)

mp_hands = mp.solutions.hands
MEDIAPIPE_CONFIG = dict(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

def _safe_xy(landmark_list, idx):
    try:
        lm = landmark_list.landmark[idx]
        return float(lm.x), float(lm.y)
    except:
        return 0.0, 0.0

def extract_hand_features(hand_landmarks):
    if hand_landmarks is None: return [0.0] * 42
    return [c for i in range(21) for c in _safe_xy(hand_landmarks, i)]

def extract_frame_features(results):
    r_hand, l_hand = [0.0] * 42, [0.0] * 42
    if results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            hand_landmarks = results.multi_hand_landmarks[idx]
            label = hand_handedness.classification[0].label
            if label == 'Right': r_hand = extract_hand_features(hand_landmarks)
            else: l_hand = extract_hand_features(hand_landmarks)
    return np.array(r_hand + l_hand, dtype=np.float32)

def process_video(filepath, hands_model):
    cap = cv2.VideoCapture(filepath)
    frames = []
    if not cap.isOpened(): return np.empty((0, FEATURE_DIM), dtype=np.float32)
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands_model.process(rgb)
        frames.append(extract_frame_features(results))
    cap.release()
    if not frames: return np.empty((0, FEATURE_DIM), dtype=np.float32)
    return np.array(frames, dtype=np.float32)

def compute_motion_energy(seq, part_slices=(RHAND_SLICE, LHAND_SLICE)):
    if seq.shape[0] < 2: return np.zeros(seq.shape[0], dtype=np.float32)
    energy = np.zeros(seq.shape[0], dtype=np.float32)
    for sl in part_slices:
        block = seq[:, sl]
        diffs = np.diff(block, axis=0)
        mag   = np.linalg.norm(diffs.reshape(diffs.shape[0], -1, 2), axis=2)
        energy[1:] += mag.mean(axis=1)
    return energy

def trim_idle_frames(seq, threshold_ratio=0.15, smooth_window=5, padding=3, min_frames=TARGET_SEQ_LEN):
    T = seq.shape[0]
    if T <= min_frames: return seq, (0, T)
    energy = compute_motion_energy(seq)
    if smooth_window > 1: energy_smooth = np.convolve(energy, np.ones(smooth_window)/smooth_window, mode="same")
    else: energy_smooth = energy
    max_e = energy_smooth.max()
    if max_e <= 1e-8: return seq, (0, T)
    active_idx = np.where(energy_smooth > threshold_ratio * max_e)[0]
    if len(active_idx) == 0: return seq, (0, T)
    start = max(0, int(active_idx[0])  - padding)
    end   = min(T, int(active_idx[-1]) + padding + 1)
    if (end - start) < min_frames: return seq, (0, T)
    return seq[start:end], (start, end)

def temporal_normalize(seq, target_len=TARGET_SEQ_LEN):
    T = seq.shape[0]
    if T == 0: return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    indices = [min(int(math.floor(i * T / target_len)), T-1) for i in range(target_len)]
    return seq[indices].astype(np.float32)

def spatial_normalize_hands(seq):
    mask = (seq != 0.0)
    valid_x = seq[:, 0::2][seq[:, 0::2] != 0.0]
    valid_y = seq[:, 1::2][seq[:, 1::2] != 0.0]
    if len(valid_x) == 0 or len(valid_y) == 0: return seq
    mean_x, mean_y = float(valid_x.mean()), float(valid_y.mean())
    range_x, range_y = float(valid_x.max() - valid_x.min()), float(valid_y.max() - valid_y.min())
    scale_x = range_x if range_x > 1e-6 else 1.0
    scale_y = range_y if range_y > 1e-6 else 1.0
    norm_seq = seq.copy()
    norm_seq[:, 0::2] = (norm_seq[:, 0::2] - mean_x) / scale_x
    norm_seq[:, 1::2] = (norm_seq[:, 1::2] - mean_y) / scale_y
    return (norm_seq * mask).astype(np.float32)

def normalize_sequence(seq):
    return spatial_normalize_hands(temporal_normalize(seq, TARGET_SEQ_LEN))

def video_to_model_input(filepath):
    with mp_hands.Hands(**MEDIAPIPE_CONFIG) as hands_model:
        raw_seq = process_video(filepath, hands_model)
    if raw_seq.shape[0] == 0:
        return None, raw_seq
    trimmed_seq, _ = trim_idle_frames(raw_seq)
    norm_seq = normalize_sequence(trimmed_seq)
    return norm_seq[np.newaxis, ...], raw_seq

class LearnedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len, self.d_model = max_len, d_model
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=d_model, embeddings_initializer="glorot_uniform", name="pos_embedding")
    def call(self, x):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        return x + self.pos_emb(positions)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg

def build_transformer_model(seq_len=TARGET_SEQ_LEN, feature_dim=FEATURE_DIM, num_classes=17, dropout_rate=0.3, seed=42):
    init   = tf.keras.initializers.GlorotUniform(seed=seed)
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), name="input")

    x = tf.keras.layers.Lambda(lambda t: t * tf.math.sqrt(tf.cast(feature_dim, tf.float32)))(inputs)
    x = LearnedPositionalEncoding(max_len=seq_len, d_model=feature_dim)(x)

    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=74, dropout=dropout_rate, kernel_initializer=init)(x, x)
    attn_out = tf.keras.layers.Dropout(dropout_rate)(attn_out)

    x = tf.keras.layers.Add()([x, attn_out])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = tf.keras.layers.Dense(feature_dim, activation="relu", kernel_initializer=init)(x)
    ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
    ffn = tf.keras.layers.Dense(feature_dim, activation=None, kernel_initializer=init)(ffn)

    x = tf.keras.layers.Add()([x, ffn])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=init)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_encoder_only")
