#!/usr/bin/env python
# coding: utf-8

# In[1]:


# We will then import UN declaration of Human Rights text data
#then view the data that we will be analysing 
import pandas as pd


# In[2]:


file_path = "un_declaration_hr_text_data.txt"
df = pd.read_csv(file_path, delimiter="\t")
print (df)


# In[3]:


#we will then import the natural language processing tool kit in order to obtain the most frequent words 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from textblob import TextBlob
from textblob import Word
import matplotlib.pyplot as plt


# In[4]:


#we will then remove all the stop words 
from nltk.corpus import stopwords


# In[5]:


words = df['Universal Declaration of Human Rights ']


# In[6]:



def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    preprocessed_text = [word for word in tokens if word.isalpha() and word not in stop_words]
    return preprocessed_text


# In[7]:


preprocessed_words = words.apply(preprocess_text)
flattened_words = [word for title in preprocessed_words for word in title]


# In[8]:


word_freq = pd.Series(flattened_words).value_counts().head(25)


# Below is the top 25 frequent terms in the document, excluding the stop words

# In[9]:


plt.figure(figsize=(16, 8))
word_freq.plot(kind='bar')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Word Frequency')
plt.show()


# Below is a word cloud of all the most frequent words used, this is excluding stop words.

# In[10]:


wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(flattened_words))
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()


# In[ ]:




