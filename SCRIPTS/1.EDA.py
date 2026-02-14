#!/usr/bin/env python
# coding: utf-8

# # This script reads in data and provides visualization for basic features in the real and fake news dataset.

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Read and View Data

# In[41]:


# Read in data
real = pd.read_csv('True.csv')

fake = pd.read_csv('Fake.csv')


# In[42]:


# View real news dataset
real.head()


# In[43]:


# Check fake news dataset for NaNs
real.isna().sum()


# In[44]:


# View fake news dataset
fake.head()


# In[45]:


# Check fake news dataset for NaNs
fake.isna().sum()


# # Create Plots

# ## Pie Charts

# In[46]:


# See what values there are for subject (real news)
real['subject'].unique() # find unique subjects
real_subject_frequencies = real['subject'].value_counts() # count unique subjects
real_subject_frequencies

# Plot value counts directly as a pie chart
real_subject_frequencies.plot(kind='pie', autopct='%1.1f%%', title='Distribution of Subjects (Real News)')
plt.ylabel('')

plt.savefig('Distribution_of_Subjects_Real.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.25
           )
plt.show()


# In[47]:


# See what values there are for subject (fake news)
fake['subject'].unique() # find unique subjects
fake_subject_frequencies = fake['subject'].value_counts() # count unique subjects
fake_subject_frequencies

# Plot value counts directly as a pie chart
fake_subject_frequencies.plot(kind='pie', autopct='%1.1f%%', title='Distribution of Subjects (Fake News)')
plt.ylabel('') 

plt.savefig('Distribution_of_Subjects_Fake.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.25
           )
plt.show()


# ## Combine Piecharts

# In[48]:


# Create a figure and a set of subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first pie chart on ax1
ax1.pie(real_subject_frequencies.values, labels=real_subject_frequencies.index, autopct='%1.1f%%')
ax1.set_title('Distribution of Subjects (Real News)')
ax1.axis('equal') # Ensure the pie charts are equal sizes

# Plot the second pie chart on ax2
ax2.pie(fake_subject_frequencies.values, labels=fake_subject_frequencies.index, autopct='%1.1f%%')
ax2.set_title('Distribution of Subjects (Fake News)')
ax2.axis('equal') # Ensure the pie charts are equal sizes

# Adjust layout and display the plots (prevent titles/labels from overlapping)
plt.tight_layout() 

# Save figure as .png and show
plt.savefig('Distribution_of_Subjects.png')
plt.show()


# ## Calculate the lengths of each headline

# In[49]:


# Find the length of each string in each cell for the real news dataset
real_len = real.apply(lambda col: col.str.len())
real_len


# In[50]:


# Find the length of each string in each cell for the fake news dataset
fake_len = fake.apply(lambda col: col.str.len())
fake_len


# ## Plot Headline Length Distribution as Histograms

# In[52]:


# Histograms showing the frequency of each length within the "title" column
plt.style.use('ggplot')

# Create figure and a set of subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axes[0].hist(real_len['title'].dropna(), bins=30, edgecolor='black')
axes[0].set_title('Real Headline Title Length\n(in Characters) Frequency')

axes[1].hist(fake_len['title'].dropna(), bins=30, edgecolor='black')
axes[1].set_title('Fake Headline Title Length\n(in Characters) Frequency')

# Add padding so that titles don't overlap
fig.tight_layout(w_pad=6)

# Save it so that GitHub doesn't crop it
fig.savefig(
    'Distribution_of_Headline_Lengths.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.25
)

plt.show()


# In[53]:


# Plot real headline title length by itself
plt.style.use('ggplot')

real_titles = real_len['title'].dropna()

fig_real, ax_real = plt.subplots(figsize=(6, 4))

ax_real.hist(real_titles, bins=30, edgecolor='black')
ax_real.set_title('Real Headline Title Length\n(in Characters) Frequency')

fig_real.tight_layout()

fig_real.savefig(
    'Real_Headline_Length_Distribution.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.25
)

plt.show()


# In[54]:


# Plot fake headline title length by itself
plt.style.use('ggplot')

fake_titles = fake_len['title'].dropna()

fig_fake, ax_fake = plt.subplots(figsize=(6, 4))

ax_fake.hist(fake_titles, bins=30, edgecolor='black')
ax_fake.set_title('Fake Headline Title Length\n(in Characters) Frequency')

fig_fake.tight_layout()

fig_fake.savefig(
    'Fake_Headline_Length_Distribution.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.25
)

plt.show()


# ## Plot Text Length Distributions as Histograms

# In[55]:


# Create log-spaced bins based on the combined range to help visualize outliers
real_text = real_len['text'].dropna()
fake_text = fake_len['text'].dropna()

min_val = min(real_text.min(), fake_text.min())
max_val = max(real_text.max(), fake_text.max())

bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)


# In[56]:


# Real news article length frequency plot
fig_real, ax_real = plt.subplots(figsize=(6, 4))

# Use log-spaced bins
ax_real.hist(real_text, bins=bins, edgecolor='black')
ax_real.set_xscale('log')
ax_real.set_title('Real Article Text Length\n(in Characters) Frequency')

fig_real.tight_layout()

fig_real.savefig(
    'Real_Text_Length_Distribution.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.25
)

plt.show()


# In[57]:


# Fake news article length frequency plot
fig_fake, ax_fake = plt.subplots(figsize=(6, 4))

# Use log-spaced bins
ax_fake.hist(fake_text, bins=bins, edgecolor='black')
ax_fake.set_xscale('log')
ax_fake.set_title('Fake Article Text Length\n(in Characters) Frequency')

fig_fake.tight_layout()

fig_fake.savefig(
    'Fake_Text_Length_Distribution.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.25
)

plt.show()


# ### Visualize Distributions of Dates Published

# In[70]:


# Clean up date column so that it's in datetime format

# Remove ordinal suffixes (real)
real['date_clean'] = real['date'].str.replace(
    r'(\d+)(st|nd|rd|th)', 
    r'\1', 
    regex=True
)

# Remove ordinal suffixes (fake)
fake['date_clean'] = fake['date'].str.replace(
    r'(\d+)(st|nd|rd|th)', 
    r'\1', 
    regex=True
)

# Convert to datetime (real)
real['date_clean'] = pd.to_datetime(
    real['date_clean'],
    errors='coerce'
)

# Convert to datetime (fake)
fake['date_clean'] = pd.to_datetime(
    fake['date_clean'],
    errors='coerce',
    format='mixed',
    dayfirst=True
)

# Convert to month/year
# For real articles
real['year_month'] = real['date_clean'].dt.to_period('M')  # e.g., 2016-12

# For fake articles
fake['year_month'] = fake['date_clean'].dt.to_period('M')


# In[71]:


# Get counts by month-year for histograms
# Real articles
real_monthly_counts = real['year_month'].value_counts().sort_index()

# Fake articles
fake_monthly_counts = fake['year_month'].value_counts().sort_index()


# In[75]:


# Plot number of real articles published by Month-Year
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10,4))

# Convert PeriodIndex to string for matplotlib
x_labels = real_monthly_counts.index.astype(str)

ax.bar(x_labels, real_monthly_counts.values, color='skyblue', edgecolor='black')
ax.set_title('Number of Real Articles Published by Month-Year')
ax.set_xlabel('Month-Year')
ax.set_ylabel('Number of Articles')

# Rotate labels so they don’t overlap
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('Real_Articles_by_Month.png', dpi=300, bbox_inches='tight')

plt.show()


# In[76]:


# Plot number of fake articles published by Month-Year
fig, ax = plt.subplots(figsize=(10,4))

# Convert PeriodIndex to string for matplotlib
x_labels = fake_monthly_counts.index.astype(str)

ax.bar(x_labels, fake_monthly_counts.values, color='salmon', edgecolor='black')
ax.set_title('Number of Fake Articles Published by Month-Year')
ax.set_xlabel('Month-Year')
ax.set_ylabel('Number of Articles')

# Rotate labels so they don’t overlap
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('Fake_Articles_by_Month.png', dpi=300, bbox_inches='tight')


plt.show()


# In[ ]:




