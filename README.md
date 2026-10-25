# SMS Spam Detection Using Machine Learning & NLP

A complete, runnable end-to-end project that classifies SMS messages as **Ham** (legitimate) or **Spam**.

Built with classical ML (TF-IDF + LinearSVC) following industry practices: proper train/test split, pipelines, stratified evaluation, model persistence, and a Streamlit web interface.

---

## Quick Start (macOS / Linux / Apple Silicon)

```bash
# 1. Go into the project
cd SMS-Spam-Detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Re-train the model
python -m src.train

# 5. Launch the web app
streamlit run app.py
```

The app will open in your browser. Paste any SMS and click **Predict**.

---

## Project Structure

```
SMS-Spam-Detection/
├── data/
│   ├── raw/SMSSpamCollection          # Original UCI file
│   └── processed/sms_cleaned.csv      # Cleaned version
├── models/
│   ├── spam_classifier.joblib         # Saved pipeline (ready to use)
│   └── model_comparison.csv
├── src/
│   ├── train.py                       # Full training + evaluation
│   └── predict.py                     # Inference helpers
├── notebooks/
│   └── 01_data_understanding.ipynb
├── app.py                             # Streamlit web app
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Original size**: 5,574 messages
- **After cleaning**: 5,171 messages (403 exact duplicates removed)
- **Class distribution** (cleaned):
  - Ham: 4,518 (87.4 %)
  - Spam: 653 (12.6 %)

---

## Model Performance (held-out test set – 20 %)

| Model               | Accuracy | Precision (Spam) | Recall (Spam) | F1 (Spam) | Train time |
|---------------------|----------|------------------|---------------|-----------|------------|
| **LinearSVC**       | 0.977    | 0.928            | 0.886         | **0.906** | 0.11 s     |
| MultinomialNB       | 0.978    | 0.991            | 0.832         | 0.905     | 0.16 s     |
| LogisticRegression  | 0.974    | 0.900            | 0.893         | 0.897     | 0.11 s     |

**Selected model**: `LinearSVC` (best F1 on spam class)

**5-fold Stratified CV F1 (spam)**: 0.934 ± 0.026

### Confusion Matrix (LinearSVC – Test set)

```
                Predicted
              Ham    Spam
Actual Ham    895      9
Actual Spam    15    116
```

---

## How to use the model in code

```python
from src.predict import predict_message

label, score = predict_message("Congratulations! You won a free iPhone. Call now!")
print(label, score)   # spam  +0.79
```

---

## Key Design Decisions

- **TF-IDF + bigrams** – captures phrases such as “free entry”, “call now”
- **`class_weight="balanced"`** – helps with the 87/13 class imbalance
- **Full Pipeline** – vectorizer and classifier are fitted together; no data leakage
- **Spam = positive class** – all metrics treat spam as the class of interest
- **Saved with joblib** – one file contains the entire fitted pipeline

---

## Limitations

- Dataset is from ~2011; modern phishing / scam language may not be fully covered
- Short or heavily abbreviated messages can be harder to classify
- False negatives (spam predicted as ham) are more dangerous than false positives
- This is **not** a production-grade spam filter for mobile carriers

---

## License & Attribution

- Dataset © Tiago A. Almeida & José María Gómez Hidalgo
- Code: MIT

---

*Trained and verified on 26 Oct 2026*
