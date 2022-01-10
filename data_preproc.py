'''

Pre-processes data provided to remove unneeded data and saves new processed data as
CSV file 

'''

# IMPORTS
import pandas as pd
import numpy as np
import re
from pathlib import Path
from nltk.corpus import stopwords
import string
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer # For stemming the sentence
from nltk.stem import SnowballStemmer # For stemming the sentence
# from contractions import contractions_dict # to solve contractions
# from autocorrect import Speller #correcting the spellings


# get data 
df = pd.read_csv('data.csv')
num_labels = df.valence.value_counts()
df = df.reindex(np.random.permutation(df.index))

# Pre-process
df = df[['tweet', 'valence']]
    
    # lowercase
df.tweet = df.tweet.apply(lambda x: x.lower())
    # remove mentions and hashtags
df.tweet = df.tweet.apply(lambda x: re.sub(r'\@\w+|\#','', x))
    # remove HyperLinks 
df.tweet = df.tweet.apply(lambda x: re.sub(r'http\S+',"",x))
    # remove punctuation  
df.tweet = df.tweet.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

   # remove stop words 
stop_words = stopwords.words('english')
df.tweet = df.tweet.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
#     # Autospell
# spell = Speller(fast=True)
# df.tweet = df.tweet.apply(lambda x: ' '.join([spell(w)for w in x.split()]))
    # Remove numbers
df.tweet = df.tweet.apply(lambda x: ' '.join([c for c in x.split() if not c.isdigit()]))
    # Lemmentization
wordnet_lemmatizer = WordNetLemmatizer()
df.tweet = df.tweet.apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word)for word in x.split()]))

    # use word stemming
# porter = PorterStemmer()
# df.tweet = df.tweet.apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))
 
    #remove bad symbols
# BAD_SYMBOLS_RE = re.compile('[/(){}\[\]\|,;^#+_@]')
# df.tweet = df.tweet.apply(lambda x: BAD_SYMBOLS_RE.sub('',x))

# save data
df.to_csv('processed_data_V2.csv', index=False)


