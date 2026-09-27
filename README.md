# Multi-Task Speaker Identification & Emotion Recognition from Speech

A deep learning project investigating whether **speaker identification** and **emotion recognition** can be learned jointly by a single shared CNN — and, more importantly, *whether they should be*. Built on the CREMA-D speech dataset.

---

## Overview

Most speech-emotion projects train a single-purpose classifier. This project instead builds **two single-task CNN baselines** (speaker-only, emotion-only) and a **multi-task CNN** with a shared convolutional backbone and two output heads, then rigorously compares them to answer one question:

> **Does sharing features between speaker identity and emotion help, or hurt, each task?**

The short answer, based on this experiment: **it hurts both, dramatically** — the shared model collapses to near-random performance on both tasks. This is a real, diagnosable negative-transfer result, backed by training curves, confusion matrices, and embedding visualizations, with an explanation for *why*. The value of this project is in that diagnosis, not in a forced "it worked!" result.

## Dataset

**[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)** (Crowd-sourced Emotional Multimodal Actors Dataset)

- 7,442 audio clips
- 91 actors, diverse ages and ethnicities
- 6 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad
- Labels encoded directly in filenames (e.g. `1001_IEO_HAP_HI.wav` → actor `1001`, emotion `HAP` = happy, intensity `HI`) — no separate label file needed

## Pipeline

```
Raw audio (.wav)
      │
      ▼
Mel-spectrogram extraction (librosa)
      │
      ▼
   ┌──────────────────────────────┐
   │     Shared CNN Backbone      │
   │  (Conv2D + BatchNorm + Pool  │
   │   ×3, GlobalAvgPool, Dense)  │
   └───────────────┬──────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Speaker ID Head        Emotion Head
 (91-way softmax)       (6-way softmax)
```

Two single-task versions of the same backbone (one head each) are trained separately as baselines for comparison.

## Exploratory Data Analysis

The notebook includes full EDA before modeling:
- Emotion class distribution (near-balanced across 6 classes)
- Speaker distribution (near-uniform, ~76–82 clips per actor across all 91 speakers)
- Clip duration distribution (mostly 2–3.5 seconds, informing the fixed preprocessing window)
- Waveform and Mel-spectrogram visualizations, one example per emotion

## Results

| Task | Single-Task Baseline | Multi-Task Model |
|---|---:|---:|
| Speaker Identification (91 classes) | **49.0%** | 1.3% |
| Emotion Classification (6 classes) | **47.5%** | 17.1% |

Both single-task baselines are reasonable and legitimate (well above the random-chance floor of ~1.1% for 91-way speaker classification and ~16.7% for 6-way emotion). The multi-task model, in contrast, collapses to **at or near random-chance accuracy on both tasks** — not a partial degradation, but a complete failure to learn a useful shared representation.

### Diagnostics (in the notebook)

- **Training curves**: single-task baselines train smoothly and converge; the multi-task model's validation metrics stay flat and noisy across all 30 epochs, never escaping the random-chance region.
- **Confusion matrices**: the multi-task emotion head skews heavily toward predicting a single dominant class rather than distinguishing between emotions; the multi-task speaker head shows no meaningful diagonal structure at all.
- **t-SNE visualization of the shared embedding space**: reveals which task (if either) the shared backbone actually organized its representation around, and how cleanly (or not) the two label sets separate in that space.

## Finding: Complete Negative Transfer

Forcing one shared CNN backbone to jointly serve both objectives produced a total training collapse rather than a partial trade-off. Two contributing factors, consistent with the diagnostics above:

1. **Conflicting feature requirements.** Speaker identification needs features that are *invariant to emotion* (the same speaker should be recognized regardless of mood); emotion recognition needs features that are *invariant to speaker identity* (the same emotion should be recognized regardless of who's talking). These are directly opposing objectives for a single shared representation, and the combined-loss gradients pulling in both directions likely prevented the backbone from settling into a useful representation for either task.

2. **Sparse per-class data.** CREMA-D provides only ~13 clips per speaker-per-emotion combination — too little signal for a shared backbone to resolve two competing objectives simultaneously, compared to the much more learnable signal each single-task model gets when it can dedicate its full capacity to one objective.

## Limitations

- **Closed-set only:** the model recognizes exactly the 91 speakers it was trained on. An unseen voice is still forced into one of the 91 known buckets, producing a meaningless prediction.
- Given the negative transfer result, **a shared multi-task model is not recommended** for this problem in its current form — two separate single-task models perform dramatically better.

## Real-World Applicability

Even the single-task baselines are best suited to **fixed-membership scenarios** — a smart-home assistant recognizing a small set of household members, or internal call-center QA over a known agent roster — where the speaker set doesn't grow. Not suitable for open-world speaker recognition, which requires an embedding/verification-based approach rather than closed-set classification.

## Future Work

- Replace the speaker softmax head with an **embedding-based approach** (e.g. triplet loss) to support open-set speaker verification for unseen speakers
- Apply **gradient-conflict mitigation** techniques (e.g. GradNorm, PCGrad) instead of a simple weighted-sum loss, and re-balance loss weights and early-stopping criteria, to test whether the collapse can be resolved rather than only diagnosed
- Heavier augmentation (pitch shift, time-stretch, noise injection) to offset the sparse per-class data

## Tech Stack

`Python` · `TensorFlow / Keras` · `librosa` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`

## Project Structure

```
├── speaker_emotion_multitask_colab.ipynb   # full notebook: EDA, preprocessing, baselines, multi-task model, evaluation
└── README.md
```

## How to Run

1. Download [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad) (the notebook includes a cell to fetch it directly via the Kaggle API)
2. Run on Google Colab or Kaggle with a GPU runtime enabled (recommended — CPU training is 50-100x slower for this pipeline)
3. Update `DATA_DIR` in the config cell to match your environment
4. Run all cells top to bottom

```bash
pip install tensorflow librosa scikit-learn pandas matplotlib seaborn tqdm
```

## Key Takeaway

This project's contribution isn't a state-of-the-art accuracy number — it's a controlled experiment showing that **naively sharing a CNN backbone across two speech tasks with conflicting objectives can produce a complete training collapse**, not just a mild trade-off, with training curves, confusion matrices, and embedding visualizations to support the diagnosis.
