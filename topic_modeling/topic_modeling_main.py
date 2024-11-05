import numpy as np
import pandas as pd
import json
import glob
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy
from nltk.corpus import stopwords

english_stopwords = stopwords.words("english")


data = pd.read_csv(r"C:\Users\alkrd\Desktop\graduation_project\the_project\preprocssed_data\cleaned_data_USA.csv")
dataFrame = data[['newsSnippet']]

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    print("lemmatizing...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    text_out = []
    for text in texts:
        doc = nlp(text)
        new_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        final = " ".join(new_text)
        text_out.append(final)
    return text_out


lemmatized_texts = lemmatization(dataFrame['newsSnippet'].astype(str).tolist())
print("lemmitized sample: ", lemmatized_texts[100])

def gen_words(texts):
    print("removing stop words...")
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_texts)
print("after removing stop words: ",data_words[100])

id2word = corpora.Dictionary(data_words)
corpus = []
print("doing some shi...")
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)
print("corpus? ", corpus[0])
