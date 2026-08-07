# Composer Identification via Deep Learning
### AAI-511 · University of San Diego · Final Project

## Project Overview

Develop a deep learning model that accurately predicts the composer of a given classical music piece. The project implements two architectures — LSTM and CNN — trained on MIDI file features extracted from compositions by well-known classical composers.

| | |
|---|---|
| **Composers** | Bach, Beethoven, Chopin, Mozart (4) |
| **Input Format** | MIDI files — feature extraction via `pretty_midi` |
| **Architectures** | LSTM (temporal note sequences) + CNN (piano-roll images) |
| **Frameworks** | TensorFlow / Keras |
| **Deliverables** | Notebook · Report (APA 7 PDF) · GitHub Repo |
| **Due Date** | August 26, 2026 |

**Methodology steps:** Data Collection → Pre-processing → Feature Extraction → Model Building → Training → Evaluation → Optimization

---

## Phase 1: Data Collection & Exploration ✅

- [x] Download and organize the MIDI dataset into composer-labeled folders
  - Structure: `data/{train,dev,test}/{bach,beethoven,chopin,mozart}/`
  - 198 files total — 166 train / 16 dev / 16 test (≈84 / 8 / 8 %)
- [x] Explore dataset distribution — file counts per composer, average duration
- [x] Validate all MIDI files are readable with `pretty_midi`; replace corrupted files
- [x] Visualize piano rolls for all 4 composers

---

## Phase 2: Data Pre-processing & Feature Extraction ✅

- [x] Extract note sequences using `pretty_midi` (music21 dropped — failed to parse several files); chords detected by grouping onsets within 50 ms → joined tokens (e.g. `C4.E4.G4`)
- [x] Build vocabulary from training data only (34,760 unique tokens; OOV → index 0, no data leakage)
- [x] Create fixed-length sliding windows of note sequences (window = 100 notes, step = 10)
- [x] Generate 2D piano-roll matrices (128 pitches × 128 frames), normalized to [0, 1]
- [x] Apply data augmentation: pitch shifting (±1, ±2 semitones) + tempo scaling (×0.9, ×1.1) → 7× CNN training set
- [x] Save preprocessed arrays to `preprocessed/lstm_data.npz` and `preprocessed/cnn_data.npz`

**Resulting array sizes**

| Representation | Train | Dev | Test |
|---|---|---|---|
| Vocabulary | 34,760 unique notes/chords | — | — |
| LSTM windows | 25,620 | 4,249 | 3,215 |
| CNN windows (base) | 16,364 | 2,340 | 2,033 |
| CNN windows (augmented) | 114,548 | 2,340 | 2,033 |

---

## Phase 3: Model Building — LSTM ✅

**Architecture:** `Embedding(34760→64) → LSTM(256) → Dropout(0.3) → LSTM(128) → Dense(64, relu) → Dense(4, softmax)`

- [x] Build LSTM model in TensorFlow (~2.76M parameters)
- [x] Categorical cross-entropy loss + Adam optimizer
- [x] EarlyStopping (patience=10, monitor val_accuracy) + ModelCheckpoint
- [x] Plot training/validation accuracy and loss curves

**Results:** Dev 57.43% · Test 55.96% · Best epoch: 9 · Stopped at epoch 19

> Limitation: severe overfitting (train ~99% vs dev ~55%) driven by the 34,760-token vocabulary — most chord tokens are too rare for the embedding layer to learn meaningful representations.

---

## Phase 4: Model Building — CNN ✅

**Architecture:** `Conv2D(32,3×3) → BN → Pool → Conv2D(64,3×3) → BN → Pool → Dropout(0.25) → GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(4, softmax)`

- [x] Build CNN model in TensorFlow (~28K parameters via GlobalAveragePooling2D)
- [x] Categorical cross-entropy loss + Adam optimizer
- [x] EarlyStopping (patience=8, monitor val_accuracy) + ModelCheckpoint
- [x] Train on 114,548 augmented piano-roll windows
- [x] Plot training/validation accuracy and loss curves

**Results:** Dev 85.90% · Test 84.36% · Best epoch: 24 · Stopped at epoch 32

> GlobalAveragePooling2D replaced Flatten to reduce parameters from 8.41M to ~28K, cutting the overfitting gap and improving test accuracy by +4.82 percentage points over the Flatten baseline.

---

## Phase 5: Evaluation & Comparison ✅

- [x] Classification reports (precision, recall, F1 per composer) for LSTM and CNN — dev and test sets
- [x] Confusion matrices: 2×2 grid (LSTM dev/test, CNN dev/test)
- [x] LSTM vs CNN model comparison table (accuracy, precision, recall, F1 macro)
- [x] Hyperparameter summary table
- [x] Bar chart: LSTM vs CNN across all metrics

**Key findings:**
- Bach: 100% precision and recall on CNN test set
- Chopin: 93.95% F1 on CNN test set
- Beethoven & Mozart most confused (~74–76% F1) — shared stylistic era
- LSTM struggled most with Chopin (38% dev F1) — wide-span arpeggios invisible to note sequences

---

## Phase 6: Report & Submission 🔲

- [ ] Add concluding markdown cell to notebook (key findings + future work)
- [ ] Export notebook as PDF or HTML
- [ ] Write APA 7 technical report (`Project_Report-Team#.pdf`)
- [ ] Push final version to GitHub

---

## Assignment References

| Assignment | Topic | Reusable For |
|---|---|---|
| A1 | Tensor basics (TF + PyTorch) | Building input tensors from MIDI features |
| A2 | ANN classification, metrics | Train/eval pipeline, classification report |
| A3 | RNN/LSTM end-to-end examples | LSTM architecture starting point |
