import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel

import spacy
from nltk.corpus import stopwords

import pyLDAvis
import pyLDAvis.gensim

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



bigrams_phrases = gensim.models.Phrases(data_words, min_count=3, threshold=80)
trigram_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=80)

bigram = gensim.models.phrases.Phraser(bigrams_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return(bigram[doc] for doc in texts)

def make_tigram(texts):
    return (trigram[bigram[doc]] for doc in texts)

data_bigrams = list(make_bigrams(data_words))
data_bigrams_trigrams = list(make_tigram(data_bigrams))

print(data_bigrams_trigrams)



id2word = corpora.Dictionary(data_bigrams_trigrams)
texts = data_bigrams_trigrams
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[0])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words = []
words_missing_in_tfidf = []

for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = []
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]  
    corpus[i] = new_bow

lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=30,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=20,
    alpha="auto"
)