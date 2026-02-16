#!/usr/bin/env python
# coding: utf-8

# In[20]:


#DS4002 Project 1 – Data Appendix Generation

#This script generates descriptive statistics and visualizations used in the project appendix to better understand the headline dataset characteristics.

#Input file: DATA/headlines_clean.csv

#Columns: title -> headline text analyzed for length and distribution
#         label -> 0 = real news, 1 = fake news

#Method: Pandas computes summary statistics
#Matplotlib/Seaborn create distribution plots and dataset visualizations

#Output: Headline length plots, label distribution plots, and dataset summary figures

#Running this script reproduces the descriptive figures reported in the project appendix


# In[21]:


# Import libraries used for reading data and plotting figures
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned analysis dataset that was used in the model
df = pd.read_csv("headlines_clean.csv")

# Preview the first few rows to verify data loaded correctly
df.head()


# In[22]:


# Check dataset dimensions (rows, columns)
df.shape


# In[23]:


# Create a new quantitative variable: number of words in each headline
# This converts text into a measurable numeric feature
df["title_length"] = df["title"].str.split().apply(len)

# Display first few calculated values
df["title_length"].head()


# In[24]:


# Generate summary statistics for headline length
# Provides count, mean, std, min, quartiles, and max
df["title_length"].describe()


# In[25]:


# Plot histogram showing distribution of headline lengths
plt.figure(figsize=(6,4))
plt.hist(df["title_length"], bins=50)

# Label axes and title for interpretation
plt.xlabel("Headline length (number of words)")
plt.ylabel("Count")
plt.title("Distribution of headline length")

plt.tight_layout()
plt.show()


# In[26]:


# Create boxplot to visualize spread and outliers in headline length
plt.figure(figsize=(5,4))
plt.boxplot(df["title_length"], vert=False)

plt.xlabel("Headline length (words)")
plt.title("Boxplot of headline length")

plt.tight_layout()
plt.show()


# In[27]:


# Count number of observations in each class (real vs fake)
df["label"].value_counts()


# In[28]:


# Calculate proportion of each class
# Helps determine if dataset is balanced
df["label"].value_counts(normalize=True)


# In[29]:


# Bar chart visualizing distribution of real vs fake headlines
plt.figure(figsize=(5,4))
df["label"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Label (0 = real, 1 = fake)")
plt.ylabel("Count")
plt.title("Distribution of headline labels")

plt.tight_layout()
plt.show()

