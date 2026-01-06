# Steam Review Helpfulness Prediction

Predicts whether a Steam review will be voted **helpful** by the community using only information available at the time the review is posted.

**Final Result:** Test F1 = **0.68**, outperforming the majority-class baseline (0.67).

## Overview

Steam reviews vary widely in quality, and community voting can be slow and inconsistent.  
This project builds a supervised machine learning model that automatically predicts review helpfulness based on review content and reviewer metadata.

## Dataset

- 100,000 filtered Steam reviews
- Balanced 50/50 helpful vs not-helpful labels
- Temporal train/validation/test split (80/10/10)
- English-only reviews using character-based language filtering

Labels are derived manually based on the amount of positive community votes, requiring a minimum level of engagement to reduce noise.

## Features

### Text Features
- **Word-level TF-IDF**
  - Unigrams + bigrams
  - Captures common phrases and structured feedback
- **Character-level TF-IDF (3–5 grams)**
  - Captures gaming slang, typos, abbreviations, and informal language
  - Provides robustness beyond word tokenization

### Metadata Features
All metadata features are available at the time the review is posted.
- Log-scaled review length
- Games owned
- Total playtime
- Number of reviews written
- Steam purchase flag
- Received-for-free flag
- Early access flag

Log scaling is applied to handle heavily skewed distributions.

## Model

- L2-regularized **Logistic Regression**
- Sparse text features concatenated with dense metadata features
- Class-weighted to handle class balance
- Decision threshold tuned on the validation set to maximize F1

The model favours interpretability, efficiency, and robustness over unnecessary complexity.

## Results

| Model | Test F1 |
|------|--------|
| Majority class baseline | 0.6667 |
| Metadata-only model | 0.667 |
| Word TF-IDF only | 0.674 |
| **Word + Char TF-IDF + Metadata** | **0.68** |

Character-level n-grams provided the largest performance gain by capturing informal and misspelled language common in Steam reviews.

## Key Takeaways

- Simple linear models outperform more complex models for this task
- Metadata is highly predictive but insufficient on its own
- Character-level text features significantly improve robustness
- Review helpfulness is inherently subjective, limiting achievable performance

## Dataset Attribution

This project uses a subset of the **100 Million+ Steam Reviews** dataset available on Kaggle.

- Source: Kaggle, *100 Million+ Steam Reviews*
- The dataset is **not included** in this repository due to size and licensing considerations
- All preprocessing, filtering, and labeling logic is implemented in `src/filter_dataset.py`

Dataset link:  
[https://www.kaggle.com/datasets/mexwell/steamreviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews)
