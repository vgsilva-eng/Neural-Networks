# Neural-Networks
### AAI-511 · Composer Identification via Deep Learning · University of San Diego

Predict the composer of a classical music piece using LSTM and CNN models trained on MIDI feature representations (Bach, Beethoven, Chopin, Mozart).

---

## Model Architectures

### LSTM — Temporal Sequence Model
Processes note sequences to learn temporal composer style patterns.

```
Input: Note Sequence (window × 1)
        ↓
Embedding Layer (vocab_size → 64)
        ↓
LSTM(256, return_sequences=True)
        ↓
Dropout(0.3)
        ↓
LSTM(128)
        ↓
Dense(64, activation='relu')
        ↓
Dense(n_composers, activation='softmax')
```

- **Loss:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Input:** Integer-encoded note sequences via sliding window

### CNN — Piano-Roll Image Model
Treats piano-roll matrices as 2D images to capture spatial pitch-time patterns.

```
Input: Piano Roll (128 × time_steps × 1)
        ↓
Conv2D(32, 3×3, relu) + BatchNormalization
        ↓
MaxPooling2D(2×2)
        ↓
Conv2D(64, 3×3, relu) + BatchNormalization
        ↓
MaxPooling2D(2×2) + Dropout(0.25)
        ↓
Flatten → Dense(128, relu) + Dropout(0.3)
        ↓
Dense(n_composers, activation='softmax')
```

- **Loss:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Input:** 2D piano-roll arrays normalized to [0, 1]

---

## Jupyter Notebook Structure

| Section | Content |
|---|---|
| **1. Setup & Imports** | Install dependencies, set random seeds, detect GPU |
| **2. Data Loading & EDA** | Load MIDI paths, class distribution chart, sample piano-roll visualizations |
| **3. Pre-processing & Feature Extraction** | Note/chord extraction + tempo (`pretty_midi`), integer encoding, LSTM sliding windows, piano-roll generation, pitch-shift + tempo-scale augmentation, file-level train/val/test split |
| **4. LSTM Model (TensorFlow)** | Model definition, training with EarlyStopping, accuracy/loss curves |
| **5. CNN Model (TensorFlow / PyTorch)** | Model definition, training with EarlyStopping, accuracy/loss curves |
| **6. Evaluation & Comparison** | Classification reports, confusion matrices, LSTM vs CNN summary table, hyperparameter analysis |
| **7. Conclusion & Future Work** | Key findings, limitations, future directions |

---

## Dataset

| Composer | Train | Dev | Test | Total |
|---|---|---|---|---|
| Bach | 42 | 4 | 4 | 50 |
| Beethoven | 42 | 4 | 4 | 50 |
| Chopin | 41 | 4 | 4 | 49 |
| Mozart | 41 | 4 | 4 | 49 |
| **Total** | **166** | **16** | **16** | **198** |

MIDI files sourced from the [midi-classic-music](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music) Kaggle dataset.

---

## Dependencies

```
pretty_midi
tensorflow
torch
scikit-learn
matplotlib
numpy
pandas
```

## Deliverables

- `Project_Report-Team#.pdf` — APA 7 technical report
- Notebook exported as PDF or HTML
- This repository with a clean commit history and `requirements.txt`
