# Composer Identification via Deep Learning
### AAI-511 · University of San Diego · Final Project

## Project Overview

Develop a deep learning model that accurately predicts the composer of a given classical music piece. The project implements two architectures — LSTM and CNN — trained on MIDI file features extracted from compositions by well-known classical composers.

| | |
|---|---|
| **Composers** | Bach, Beethoven, Chopin, Mozart, Schubert (5+) |
| **Input Format** | MIDI files — feature extraction via `music21` / `pretty_midi` |
| **Architectures** | LSTM (temporal note sequences) · CNN (piano-roll as image) |
| **Frameworks** | TensorFlow · PyTorch |
| **Deliverables** | Notebook · Report (APA 7 PDF) · GitHub Repo |
| **Due Date** | August 26, 2026 |

**Methodology steps:** Data Collection → Pre-processing → Feature Extraction → Model Building → Training → Evaluation → Optimization

---

## Project Phases — Composer Identification via Deep Learning

## Phase 1: Data Collection & Exploration

- [ ] Download and organize the provided MIDI dataset into composer-labeled folders
  - Expected structure: `data/bach/`, `data/beethoven/`, `data/chopin/`, `data/mozart/`, `data/schubert/`
- [ ] Explore dataset distribution — file counts per composer, average duration
- [ ] Validate all MIDI files are readable with `pretty_midi` or `music21`; flag and remove corrupted files
- [ ] Visualize a few MIDI piano rolls to build intuition for the data

---

## Phase 2: Data Pre-processing & Feature Extraction

- [ ] Extract note sequences (pitch, duration, offset) using `music21`
- [ ] Build vocabulary of unique notes → encode as integers (needed for LSTM embedding layer)
- [ ] Create fixed-length sliding windows of note sequences (e.g., window = 100 notes → composer label)
- [ ] For CNN: generate 2D piano-roll matrices (128 pitches × time steps), normalize to [0, 1]
- [ ] Apply data augmentation: pitch shifting, tempo scaling
- [ ] Train / Validation / Test split — 70 / 15 / 15, stratified by composer

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
