#!/usr/bin/env python
# coding: utf-8

# DS4002 Project 1 – Fake News Classification Model
# 
# This script trains and evaluates a machine learning model that classifies
# news headlines as real (0) or fake (1).
# 
# Input file:
# DATA/headlines_clean.csv
# 
# Columns:
# title  -> headline text used as predictor
# label  -> 0 = real news, 1 = fake news
# 
# Method:
# TF-IDF converts text into numerical features
# Logistic Regression performs classification
# 
# Output:
# Accuracy, Precision, Recall, F1 Score
# Classification report
# Confusion matrix
# 
# Running this script reproduces the model results reported in the project.

# In[13]:


# import libraries used for data handling and machine learning
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


# In[14]:


# load Kaggle datasets and assign classification labels
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

# combine both datasets into one dataframe
df = pd.concat([fake, true], ignore_index=True)

# keep only headline text and label
df = df[["title", "label"]]

# remove missing or empty headlines
df["title"] = df["title"].astype(str)
df = df[df["title"].str.strip() != ""]

# display dataset size and class counts
print(df.shape)
print(df["label"].value_counts())

df.head()


# In[15]:


# save cleaned dataset used for modeling
df.to_csv("headlines_clean.csv", index=False)

# reload dataset to confirm structure
clean = pd.read_csv("headlines_clean.csv")
clean.columns, clean["label"].value_counts()


# In[16]:


# split data into training and testing sets with balanced labels
X = clean["title"]
y = clean["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# create TF-IDF + Logistic Regression model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("lr", LogisticRegression(max_iter=1000))
])

# train the model
model.fit(X_train, y_train);


# In[17]:


# generate predictions and compute evaluation metrics
best_model = model

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 score:  {f1:.4f}")

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))


# In[18]:


# create confusion matrix to show prediction outcomes
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Real", "Actual Fake"],
    columns=["Predicted Real", "Predicted Fake"]
)

print("Confusion matrix:")
print(cm_df)

