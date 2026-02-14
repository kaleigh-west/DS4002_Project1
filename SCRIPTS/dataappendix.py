#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("headlines_clean.csv")

df.head()


# In[10]:


df.shape


# In[11]:


df["title_length"] = df["title"].str.split().apply(len)
df["title_length"].head()


# In[12]:


df["title_length"].describe()


# In[13]:


plt.figure(figsize=(6,4))
plt.hist(df["title_length"], bins=50)
plt.xlabel("Headline length (number of words)")
plt.ylabel("Count")
plt.title("Distribution of headline length")
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(5,4))
plt.boxplot(df["title_length"], vert=False)
plt.xlabel("Headline length (words)")
plt.title("Boxplot of headline length")
plt.tight_layout()
plt.show()


# In[15]:


df["label"].value_counts()


# In[16]:


df["label"].value_counts(normalize=True)


# In[17]:


plt.figure(figsize=(5,4))
df["label"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Label (0 = real, 1 = fake)")
plt.ylabel("Count")
plt.title("Distribution of headline labels")
plt.tight_layout()
plt.show()

