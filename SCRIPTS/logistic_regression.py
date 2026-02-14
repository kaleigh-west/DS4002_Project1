#!/usr/bin/env python
# coding: utf-8

# In[87]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


# In[88]:


fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true], ignore_index=True)

df = df[["title", "label"]]

df["title"] = df["title"].astype(str)
df = df[df["title"].str.strip() != ""]

print(df.shape)
print(df["label"].value_counts())

df.head()


# In[89]:


df.to_csv("headlines_clean.csv", index=False)


# In[90]:


clean = pd.read_csv("headlines_clean.csv")
clean.columns, clean["label"].value_counts()


# In[91]:


X = clean["title"]
y = clean["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("lr", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train);


# In[92]:


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

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Real", "Actual Fake"],
    columns=["Predicted Real", "Predicted Fake"]
)

print("Confusion matrix:")
cm_df

