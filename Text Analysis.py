# %% [markdown]
# ### Importing Required Libraries

# %%
import pandas as pd
import requests
import bs4 as bfs
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
import csv
import numpy as np

# %%
#Download the punkt package for tokenization
nltk.download('punkt')

# %% [markdown]
# ### Reading Excel File

# %%
ip_list= pd.read_excel("C:/Users/DELL/Desktop/Blackcoffer/Input.xlsx")
max_row, max_col = ip_list.shape
print(max_row)

pd.set_option('display.max_colwidth',100)
ip_list.head()

# %%
ip_list.URL.head()

# %%
li= [url for url in ip_list['URL']]
li

# %% [markdown]
# 

# %% [markdown]
# ### Parsing URLs

# %%
text= []
for url in li:
    text.append(requests.get(url,headers={"User-Agent": "XY"}))

# %%
for i in range(len(text)):
  text[i] = bfs.BeautifulSoup(text[i].content,'html.parser')

# %%
text

# %%
articles = []
for text in text:
  articles.append(text.find(attrs= {"class":"td-post-content"}).text)

# %%
for i in range(len(articles)):
  articles[i]= articles[i].replace('\n','')

# %%
articles

# %% [markdown]
# ### Importing StopWords

# %%
with open("C:/Users/DELL/Desktop/Blackcoffer/StopWords/StopWords.txt",'r') as f:
    stop_words = f.read()

stop_words = stop_words.split('\n')
print(f'Total number of Stop Words are {len(stop_words)}')

# %% [markdown]
# ### Tokenization

# %%
sentences = []
for article in articles:
  sentences.append(len(sent_tokenize(article)))

# %%
cleaned_articles = [' ']*len(articles)

# %%
for i in range(len(articles)):
  for w in stop_words:
    cleaned_articles[i]= articles[i].replace(' '+w+' ',' ').replace('?',' ').replace('.',' ').replace(',',' ').replace('!',' ')
    #removing non-word characters

# %%
cleaned_articles

# %%
words = []
for article in articles:
  words.append(len(word_tokenize(article)))

# %%
len(words)

# %%
words_cleaned = []
for article in cleaned_articles:
  words_cleaned.append(len(word_tokenize(article)))

# %%
#ip_list['Total Length'] = ip_list['cleaned Articles'].map(lambda x: len(x))

# %% [markdown]
# ### Importing +ve and -ve Words

# %%
#positive_words = pd.read_csv("C:/Users/DELL/Desktop/Blackcoffer/MasterDictionary/positive-words.txt", sep=" ")
#len(positive_words)

# %%
#negative_words = pd.read_csv("C:/Users/DELL/Desktop/Blackcoffer/MasterDictionary/negative-words.txt", sep=" ", encoding = "ISO-8859-1")
#len(negative_words)

# %%
file = open('C:/Users/DELL/Desktop/Blackcoffer/MasterDictionary/negative-words.txt', 'r')
negative_words = file.read().split()
file = open('C:/Users/DELL/Desktop/Blackcoffer/MasterDictionary/positive-words.txt', 'r')
positive_words = file.read().split()

# %%
#positive_words.head()
print(len(positive_words))
print(positive_words[:5])

# %%
#negative_words.head()
print(len(negative_words))
print(negative_words[:5])

# %%
ip_list['Cleaned Articles'] = cleaned_articles

# %%
ip_list

# %%
#mapping +ve & -ve words to articles in DataFrame
num_pos = ip_list['Cleaned Articles'].map(lambda x: len([i for i in x if i in positive_words]))
ip_list['positive_count'] = num_pos
num_neg = ip_list['Cleaned Articles'].map(lambda x: len([i for i in x if i in negative_words]))
ip_list['negative_count'] = num_neg

# %% [markdown]
# ### Calculating the Required Columns

# %%
#Positive Score
positive_score= [0]*len(articles)
for i in range(len(articles)):
  for word in positive_words:
    for letter in cleaned_articles[i].lower().split(' '):
      if letter==word:
        positive_score[i]+=1
        

# %%
#Negative Score

negative_score= [0]*len(articles)
for i in range(len(articles)):
  for word in negative_words:
    for letter in cleaned_articles[i].lower().split(' '):
      if letter==word:
        negative_score[i]+=1

# %%
words_cleaned = np.array(words_cleaned)
sentences = np.array(sentences)

# %%
ip_list['POSITIVE SCORE'] = positive_score
ip_list['NEGATIVE SCORE'] = negative_score

# %%
positive_score, negative_score

# %%
#Polarity Score
ip_list['POLARITY SCORE'] = (ip_list['POSITIVE SCORE']-ip_list['NEGATIVE SCORE'])/ ((ip_list['POSITIVE SCORE'] +ip_list['NEGATIVE SCORE']) + 0.000001)

# %%
#Subjectivity Score
ip_list['SUBJECTIVITY SCORE'] = (ip_list['POSITIVE SCORE'] + ip_list['NEGATIVE SCORE'])/( (words_cleaned) + 0.000001)

# %%
#Avg Sentence Length
ip_list['AVG SENTENCE LENGTH'] = np.array(words)/np.array(sentences)

# %%
complex_words = []
syllable_counts = []

# %%
for article in articles:
  syllable_count=0
  d=article.split()
  ans=0
  for word in d:
    count=0
    for i in range(len(word)):
      if(word[i]=='a' or word[i]=='e' or word[i] =='i' or word[i] == 'o' or word[i] == 'u'):
           count+=1
           #print(words[i])
      if(i==len(word)-2 and (word[i]=='e' and word[i+1]=='d')): #ignoring the words ending with 'ed' & 'es'
        count-=1;
      if(i==len(word)-2 and (word[i]=='e' and word[i]=='s')):
        count-=1;
    syllable_count+=count    
    if(count>2):
        ans+=1
  syllable_counts.append(syllable_count)
  complex_words.append(ans)

# %%
print(syllable_count)
print(len(complex_words))

# %%
# % of Complex Words
ip_list['PERCENTAGE OF COMPLEX WORDS'] = np.array(complex_words)/np.array(words)

# %%
#Fog Index
ip_list['FOG INDEX'] = 0.4 * (ip_list['AVG SENTENCE LENGTH'] + ip_list['PERCENTAGE OF COMPLEX WORDS'])

# %%
#Avg number of Words per Sentences
ip_list['AVG NUMBER OF WORDS PER SENTENCES'] = ip_list['AVG SENTENCE LENGTH']

# %%
#Complex Word Count
ip_list['COMPLEX WORD COUNT'] = complex_words

# %%
#Word Count
ip_list['WORD COUNT'] = words

# %%
#Syllable per Word
ip_list['SYLLABLE PER WORD'] = np.array(syllable_counts)/np.array(words)

# %%
#Personal Pronouns
personal_nouns = []
personal_noun =['I', 'we','my', 'ours','and' 'us','My','We','Ours','Us','And'] 
for article in articles:
  ans=0
  for word in article:
    if word in personal_noun:
      ans+=1
  personal_nouns.append(ans)

# %%
ip_list['PERSONAL PRONOUN'] = personal_nouns
#since all pronouns were cleared when clearing the stop words.

# %%
total_characters = []
for article in articles:
  characters = 0
  for word in article.split():
    characters+=len(word)
  total_characters.append(characters)  

# %%
#Avg Word Length
ip_list['AVG WORD LENGTH'] = np.array(total_characters)/np.array(words)

# %% [markdown]
# ### Listing Output

# %%
#Deleting the not required Columns
#ip_list.drop(labels=['Cleaned Articles', 'positive_count', 'negative_count'], axis=1, errors='ignore')

del ip_list['Cleaned Articles']
del ip_list['positive_count']
del ip_list['negative_count']

# %%
ip_list

# %%
articles

# %% [markdown]
# ### Appending to the Output Sheet

# %%
writer = pd.ExcelWriter("C:/Users/DELL/Desktop/Blackcoffer/Assignment/Output Data Structure.xlsx")
ip_list.to_excel(writer, sheet_name='Sheet1')
writer.save()

# %%



