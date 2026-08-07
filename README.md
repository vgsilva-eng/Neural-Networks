# Neural-Networks
### AAI-511 · Composer Identification via Deep Learning · University of San Diego

Predict the composer of a classical music piece using LSTM and CNN models trained on MIDI feature representations (Bach, Beethoven, Chopin, Mozart).

---

## Results

| Model | Dev Accuracy | Test Accuracy | Parameters |
|---|---|---|---|
| LSTM | 55.26% | 55.05% | ~2.76M |
| **CNN** | **85.90%** | **84.36%** | **~28K** |

**CNN per-composer test F1:** Bach 100% · Chopin 93.95% · Beethoven 74.55% · Mozart 74.39%

---

## Model Architectures

### LSTM — Temporal Sequence Model
Processes integer-encoded note sequences to learn temporal composer style patterns.

```
Input: Note Sequence (100 tokens)
        ↓
Embedding (34,760 → 64)
        ↓
LSTM(256, return_sequences=True)
        ↓
Dropout(0.3)
        ↓
LSTM(128)
        ↓
Dense(64, relu)
        ↓
Dense(4, softmax)
```

- **Loss:** Categorical Cross-Entropy · **Optimizer:** Adam · **Best epoch:** 9

### CNN — Piano-Roll Image Model
Treats piano-roll matrices as 2D grayscale images to capture spatial pitch-time patterns.

```
Input: Piano Roll (128 × 128 × 1)
        ↓
Conv2D(32, 3×3, relu) + BatchNormalization + MaxPooling2D(2×2)
        ↓
Conv2D(64, 3×3, relu) + BatchNormalization + MaxPooling2D(2×2) + Dropout(0.25)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, relu) + Dropout(0.3)
        ↓
Dense(4, softmax)
```

- **Loss:** Categorical Cross-Entropy · **Optimizer:** Adam · **Best epoch:** 24
- `GlobalAveragePooling2D` reduces parameters from 8.41M to ~28K versus a `Flatten` baseline, improving test accuracy by +4.82 pp

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

## Preprocessing

- **LSTM:** note/chord extraction → vocabulary (34,760 tokens, train only) → sliding windows (size=100, step=10)
- **CNN:** piano-roll generation (128×128 frames, fs=10) → augmentation: ±1,±2 semitone pitch shifts + ×0.9,×1.1 tempo scales → **114,548 training windows**

---

## Notebook Structure

| Section | Content |
|---|---|
| **Phase 1** | Data loading, file counts, duration stats, piano-roll visualizations |
| **Phase 2** | Note extraction, vocabulary, LSTM windows, CNN piano-rolls, augmentation |
| **Phase 3** | LSTM model — training, EarlyStopping, accuracy/loss curves |
| **Phase 4** | CNN model — training, EarlyStopping, accuracy/loss curves |
| **Phase 5** | Classification reports, confusion matrices, LSTM vs CNN comparison |
| **Phase 6** | Conclusion and future work |

---

## Dependencies

```
pretty_midi
tensorflow
scikit-learn
matplotlib
numpy
pandas
```

## Deliverables

- `Project_Report-Team#.pdf` — APA 7 technical report
- Notebook exported as PDF or HTML
- This repository with clean commit history
