# Composer Identification via Deep Learning
### AAI-511 · University of San Diego · Final Project

## Project Overview

Develop a deep learning model that accurately predicts the composer of a given classical music piece. The project implements two architectures — LSTM and CNN — trained on MIDI file features extracted from compositions by well-known classical composers.

| | |
|---|---|
| **Composers** | Bach, Beethoven, Chopin, Mozart (4) |
| **Input Format** | MIDI files — feature extraction via `pretty_midi` |
| **Architectures** | LSTM (temporal note sequences) · CNN (piano-roll as image) |
| **Frameworks** | TensorFlow · PyTorch |
| **Deliverables** | Notebook · Report (APA 7 PDF) · GitHub Repo |
| **Due Date** | August 26, 2026 |

**Methodology steps:** Data Collection → Pre-processing → Feature Extraction → Model Building → Training → Evaluation → Optimization

---

## Project Phases — Composer Identification via Deep Learning

## Phase 1: Data Collection & Exploration

- [x] Download and organize the MIDI dataset into composer-labeled folders
  - Structure: `data/{train,dev,test}/{bach,beethoven,chopin,mozart}/`
  - 198 files total — 166 train / 16 dev / 16 test (≈84 / 8 / 8 %)
- [x] Explore dataset distribution — file counts per composer, average duration
- [x] Validate all MIDI files are readable with `pretty_midi`; replace corrupted files
- [x] Visualize piano rolls for all 4 composers

---

## Phase 2: Data Pre-processing & Feature Extraction

- [x] Extract note sequences using `pretty_midi` (music21 dropped — it failed to parse several files); chords detected by grouping onsets within 50 ms → joined tokens (e.g. `C4.E4.G4`)
- [x] Build vocabulary of unique notes → encode as integers (needed for LSTM embedding layer)
- [x] Create fixed-length sliding windows of note sequences (window = 100 notes, step = 10 → composer label)
- [x] For CNN: generate 2D piano-roll matrices (128 pitches × time steps), normalize to [0, 1]
- [x] Apply data augmentation: pitch shifting
- [x] Train / Validation / Test split kept at the pre-existing folder split (166 / 16 / 16 ≈ 84 / 8 / 8); split at **file level** so no windows leak across sets

---

## Phase 2b: Feature & Augmentation Enhancements

Additions layered on top of Phase 2 to strengthen the stylistic signal and CNN generalization.

- [x] Extract mean **tempo (BPM)** per file via `extract_midi_features()` and record it per composer (`tempo_records`). Finding: most files carry no tempo-change events and fall back to 120 BPM (e.g. all 42 Bach files = 120.0; Beethoven 118.4, Chopin 117.6, Mozart 110.0), so extracted tempo is a **weak** discriminator in this dataset and is not used as a headline feature in the report
- [x] Add **tempo-scaling** augmentation (0.9×, 1.1×) to the CNN training set alongside the existing pitch shifts, applied over a captured base so already-augmented windows are never re-augmented
- [x] Save preprocessed arrays to `preprocessed/lstm_data.npz` and `preprocessed/cnn_data.npz`

**Resulting array sizes**

| Representation | Train | Dev | Test |
|---|---|---|---|
| Vocabulary | 34,760 unique notes/chords (train only, OOV → 0) | — | — |
| LSTM windows | 25,620 | 4,249 | 3,215 |
| CNN windows (base) | 16,364 | 2,340 | 2,033 |
| CNN windows (augmented) | 114,548 (7× = base + 4 pitch shifts + 2 tempo scales) | 2,340 | 2,033 |

> Note: the small test set (4 files/composer) is a real limitation — Phase 5 reports both window-level and file-level majority-vote accuracy and flags it in the Limitations section.

---

## Phase 3: Model Building — LSTM

Architecture: `Embedding → LSTM(256) → Dropout(0.3) → LSTM(128) → Dense(64, relu) → Softmax`

- [ ] Build LSTM model in TensorFlow (and PyTorch as reference)
- [ ] Use categorical cross-entropy loss + Adam optimizer
- [ ] Train with EarlyStopping (monitor val_accuracy, patience=10) and ModelCheckpoint
- [ ] Plot training/validation accuracy and loss curves

---

## Phase 4: Model Building — CNN

Architecture: `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Softmax`

- [ ] Build 2D CNN treating piano-roll matrices as grayscale images
- [ ] Experiment with filter sizes: 3×3 and 5×5
- [ ] Add BatchNormalization and Dropout(0.3) for regularization
- [ ] Train and plot convergence curves

---

## Phase 5: Evaluation & Optimization

- [ ] Evaluate both models: Accuracy, Precision, Recall, F1-score (`sklearn.metrics.classification_report`)
- [ ] Plot confusion matrices for LSTM and CNN side-by-side
- [ ] Hyperparameter tuning: learning rate, batch size, LSTM units, dropout rate
- [ ] Compare LSTM vs CNN in a summary table
- [ ] Error analysis: which composers are most confused with each other?

---

## Phase 6: Report & Submission

- [ ] Write APA 7 technical report: Introduction, Methodology, Results, Conclusion
  - File naming: `Project_Report-Team#.pdf`
- [ ] Add concluding markdown cell to notebook summarizing findings and future work
- [ ] Export notebook as PDF or HTML
- [ ] Push all code to GitHub repo with clear README and `requirements.txt`

---

## Assignment References

| Assignment | Topic | Reusable For |
|---|---|---|
| A1 | Tensor basics (TF + PyTorch) | Building input tensors from MIDI features |
| A2 | ANN classification, metrics | Train/eval pipeline, classification report |
| A3 | RNN/LSTM end-to-end examples | LSTM architecture starting point |
