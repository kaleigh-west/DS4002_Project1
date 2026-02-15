#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries used for reading data and plotting figures
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned analysis dataset that was used in the model
df = pd.read_csv("headlines_clean.csv")

# Preview the first few rows to verify data loaded correctly
df.head()


# In[3]:


# Check dataset dimensions (rows, columns)
df.shape


# In[4]:


# Create a new quantitative variable: number of words in each headline
# This converts text into a measurable numeric feature
df["title_length"] = df["title"].str.split().apply(len)

# Display first few calculated values
df["title_length"].head()


# In[5]:


# Generate summary statistics for headline length
# Provides count, mean, std, min, quartiles, and max
df["title_length"].describe()


# In[6]:


# Plot histogram showing distribution of headline lengths
plt.figure(figsize=(6,4))
plt.hist(df["title_length"], bins=50)

# Label axes and title for interpretation
plt.xlabel("Headline length (number of words)")
plt.ylabel("Count")
plt.title("Distribution of headline length")

plt.tight_layout()
plt.show()


# In[7]:


# Create boxplot to visualize spread and outliers in headline length
plt.figure(figsize=(5,4))
plt.boxplot(df["title_length"], vert=False)

plt.xlabel("Headline length (words)")
plt.title("Boxplot of headline length")

plt.tight_layout()
plt.show()


# In[8]:


# Count number of observations in each class (real vs fake)
df["label"].value_counts()


# In[9]:


# Calculate proportion of each class
# Helps determine if dataset is balanced
df["label"].value_counts(normalize=True)


# In[10]:


# Bar chart visualizing distribution of real vs fake headlines
plt.figure(figsize=(5,4))
df["label"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Label (0 = real, 1 = fake)")
plt.ylabel("Count")
plt.title("Distribution of headline labels")

plt.tight_layout()
plt.show()

