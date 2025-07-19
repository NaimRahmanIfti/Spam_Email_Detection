# =============================
# 📧 Spam Email Detection Project
# Full Python Script Version
# =============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For data visualization
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore
# ========== 1. Load Data ==========

# File paths
file_path1 = "/Users/naim/Desktop/Spam Email Detection/old_spam_email_dataset.csv"
file_path2 = "/Users/naim/Desktop/Spam Email Detection/new_spam_email_dataset.csv"

# Read CSVs
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Rename necessary columns
df1 = df1.rename(columns={'Body': 'text', 'Label': 'spam'})
df2 = df2.rename(columns={'body_final_modified': 'text', 'Label': 'spam'})

# Drop missing values
df1.dropna(inplace=True)
df2.dropna(inplace=True)

# Select only necessary columns
df1 = df1[['text', 'spam']]
df2 = df2[['text', 'spam']]

# ========== 2. Preprocessing ==========

# Remove duplicate rows
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

# Text features
for df in [df1, df2]:
    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(x.split()))
    df['num_sentences'] = df['text'].apply(lambda x: x.count('.') + 1)

# ========== 3. EDA (optional plotting section) ==========

# Plotting functions (skip in .py if you don't need graphs)

# ========== 4. Text Cleaning ==========

stopwords_list = ['the', 'to', 'and', 'of', 'a', 'in', 'for', 'is', 'that', 'on', 'it',
                  'with', 'as', 'be', 'at', 'this', 'by', 'an', 'are', 'was', 'from', 'or', 
                  'free', 'offer', 'win', 'prize']

def simple_transform(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords_list]
    return ' '.join(cleaned_words)

# Apply transformation
df1['transformed_text'] = df1['text'].apply(simple_transform)
df2['transformed_text'] = df2['text'].apply(simple_transform)

# ========== 5. Vectorization ==========

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Old dataset
X1 = tfidf.fit_transform(df1['transformed_text']).toarray()
y1 = df1['spam']

# New dataset
X2 = tfidf.fit_transform(df2['transformed_text']).toarray()
y2 = df2['spam']

# ========== 6. Feature Selection (Mutual Information) ==========

mi_scores1 = mutual_info_classif(X1, y1)
mi_scores2 = mutual_info_classif(X2, y2)

feature_names1 = tfidf.get_feature_names_out()
feature_names2 = tfidf.get_feature_names_out()

# ========== 7. Train-Test Split ==========

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# ========== 8. Train Models ==========

# Models for old dataset
dt1 = DecisionTreeClassifier()
dt1.fit(X1_train, y1_train)
y1_pred_dt = dt1.predict(X1_test)

rf1 = RandomForestClassifier(n_estimators=100)
rf1.fit(X1_train, y1_train)
y1_pred_rf = rf1.predict(X1_test)

# Models for new dataset
dt2 = DecisionTreeClassifier()
dt2.fit(X2_train, y2_train)
y2_pred_dt = dt2.predict(X2_test)

rf2 = RandomForestClassifier(n_estimators=100)
rf2.fit(X2_train, y2_train)
y2_pred_rf = rf2.predict(X2_test)

# ========== 9. Evaluation ==========

print("\n=== Old Emails Dataset Results ===")
print("\nDecision Tree Classifier Results:")
print(classification_report(y1_test, y1_pred_dt))

print("\nRandom Forest Classifier Results:")
print(classification_report(y1_test, y1_pred_rf))

print("\n=== New Emails Dataset Results ===")
print("\nDecision Tree Classifier Results:")
print(classification_report(y2_test, y2_pred_dt))

print("\nRandom Forest Classifier Results:")
print(classification_report(y2_test, y2_pred_rf))

# =============================
# END
# =============================
