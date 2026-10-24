# fake-profile-detection
End-to-end ML &amp; NLP pipeline to detect fake social media profiles using Random Forest, XGBoost, and DistilBERT.
# Detection of Fake Profiles on Social Media Platforms

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to identify automated bot accounts and fraudulent profiles on social media platforms using behavioral metadata and text features.

## 📌 Project Overview
Social media platforms face significant challenges with spam bots and fake accounts. This project benchmarks classical ensemble learning algorithms against a fine-tuned deep learning transformer architecture to classify authentic vs. fraudulent user profiles with high precision.

## 🛠️ Tech Stack & Models
* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning (Ensemble):** Random Forest, XGBoost (`scikit-learn`, `xgboost`)
* **Deep Learning & NLP:** DistilBERT (`transformers`, Deep Learning)

## 🚀 Key Workflow
1. **Data Preprocessing:** Handled missing values, encoded categorical metadata, and addressed class imbalance across profile attributes.
2. **Feature Engineering & NLP:** Extracted behavioral metrics and tokenized profile text data for sequence classification.
3. **Model Training & Benchmarking:** Trained Random Forest, XGBoost, and DistilBERT models to compare classical tabular performance against contextual embeddings.
4. **Evaluation:** Assessed model generalization using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 👤 Author
**Armoghan Ejaz**  
B.Tech Computer Science & Engineering (AI & ML) — Sikkim Manipal Institute of Technology (SMIT)
