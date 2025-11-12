# goodreads-lakehouse-lab4-60104647
# Lab 4: Text Feature Engineering on Azure Databricks

## 🧠 Overview
This lab builds upon the previous labs (Lab 1–3) in the Goodreads data pipeline project.  
The focus here is on **text feature extraction and engineering**, transforming raw textual data from the *Gold layer* into meaningful, machine-readable numerical features.  
These features will later be used for **feature selection and model training** in the next lab.

---

## 🎯 Objectives
By the end of this lab, we aim to:
- Clean and normalize raw text data from Goodreads reviews.
- Engineer **basic linguistic**, **sentiment**, and **TF-IDF** features.
- Understand how preprocessing affects downstream machine-learning performance.
- Save a structured dataset of engineered features back into the **Gold layer** for modeling.

---

## 🧩 Key Tasks and Implementation

### 1️⃣ Loading the Gold Layer
- Accessed curated Goodreads dataset from `gold/features_v1` (or `feature_v2/train`).
- Verified schema and previewed columns like:
  - `review_id`, `book_id`, `author_name`, `review_text`, `rating`, `date_updated`, `review_length`, etc.

### 2️⃣ Text Cleaning and Normalization
Applied multiple text preprocessing steps using **PySpark**:
- Converted all text to lowercase.
- Removed punctuation and extra spaces.
- Replaced URLs, numbers, and emojis using `re` and `emoji` libraries.
- Filtered out empty or very short reviews (< 10 characters).
- Registered the cleaning logic as a **UDF** and created a `cleaned_review_text` column.

### 3️⃣ Extracting Text-Based Features
#### a. Basic Features
- `review_length_words`: number of words per review.  
- `review_length_chars`: number of characters per review.

#### b. Sentiment Features
Implemented sentiment analysis using **NLTK VADER**:
- Computed:
  - `sentiment_pos` (positive polarity),
  - `sentiment_neg` (negative polarity),
  - `sentiment_neu` (neutral polarity),
  - `sentiment_compound` (overall sentiment score).
- Integrated scores into the DataFrame `df_features`.

#### c. TF-IDF Features
Used **Spark MLlib** or a **safe PySpark alternative** to:
- Tokenize and clean text.
- Compute Term Frequency–Inverse Document Frequency (TF-IDF) to quantify word importance.
- Produced a new column `tfidf_features` or `avg_tfidf_score`.

---

## 🧰 Technologies and Libraries
- **Azure Databricks** (Spark environment)  
- **PySpark** (DataFrame operations and MLlib feature extraction)  
- **NLTK** (VADER sentiment analysis)  
- **Regex / re** and **emoji** (text cleaning)  
- **pandas** (used temporarily in local vectorization)  
- **scikit-learn** (for TF-IDF testing when resources allow)

---

## 💾 Output and Storage
- Cleaned and feature-engineered dataset stored as a **Delta table** in the Gold layer:
abfss://lakehouse@goodreadsreviewsgen2.dfs.core.windows.net/gold/features_v2/
- Final dataset contains:
- Original metadata (`book_id`, `author_name`, etc.)
- Engineered text features (`sentiment_*`, `tfidf_features`, `review_length_*`)

---

## 📊 Expected Results
- DataFrame with multiple engineered columns per review.
- Ready for use in **feature selection and modeling** (Lab 5).
- Verified successful schema display and sample rows in Databricks.

---

## 🧪 Example Code Snippets

### Load Data
```python
gold_train_path = "abfss://lakehouse@goodreadsreviewsgen2.dfs.core.windows.net/gold/features_v1"
df = spark.read.format("delta").load(gold_train_path)
display(df.limit(5))


##Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def get_sentiment_score(text, key):
    if not text:
        return None
    scores = sid.polarity_scores(text)
    return float(scores[key])

## TF-IDF Extraction (Spark Version)

from pyspark.ml.feature import CountVectorizer, IDF
cv = CountVectorizer(inputCol="words", outputCol="raw_features", vocabSize=1000)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
