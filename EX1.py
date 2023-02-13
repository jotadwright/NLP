import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
from scipy.stats import kendalltau

import spacy
import en_core_web_sm
nlp_es = spacy.load("es_core_news_sm")
import es_core_news_sm


def clean(text):
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('\n', '', text)

    return text


def filt(doc):
    if type(doc) is list:
        words = [token for token in doc if (token.is_alpha and not token.is_punct)]
        words = [token.lower() for token in words]
    else:

        words = [token.text for token in doc if (token.is_alpha and not token.is_punct)]
        words = [token.lower() for token in words]

    return words


def extract(df):
    cor_en = df['EN'].apply(lambda x: re.sub('<.*?>', '', x))
    cor_es = df['ES'].apply(lambda x: re.sub('<.*?>', '', x))

    cor_en = ' '.join(cor_en.to_list())
    cor_es = ' '.join(cor_es.to_list())

    return cor_en, cor_es


def dict2df(dic):
    # Put word frequencies in a dataframe, sort by descending frequency
    df = pd.DataFrame.from_dict(dic, orient='index').reset_index()
    df = df.rename({'index': 'word', 0: 'frequency'}, axis=1)
    df = df.sort_values(by='frequency', ascending=False, ignore_index=True)

    # Get character length of each word
    df['length'] = df['word'].apply(len)

    return df


def kendcorr(X, Y):
    corr, p = kendalltau(X, Y)

    return corr, p



# Open and read txt files
en_text = open('texts/new_poetry.txt', "r")
en_text = en_text.read()

es_text = open('texts/antologia.txt', 'r')
es_text = es_text.read()

# Get corpus samples as baseline
parallel_corpus = pd.read_csv('texts/parallel.csv', sep='\t')
en_corpus, es_corpus = extract(parallel_corpus)

# Clean text before tokenisation
en_text = clean(en_text)
es_text = clean(es_text)

# Load spacy models and process texts
nlp_en = en_core_web_sm.load()
nlp_es = es_core_news_sm.load()

en_doc = nlp_en(en_text)
en_cor_doc = nlp_en(en_corpus)
es_doc = nlp_es(es_text)
es_cor_doc = nlp_es(es_corpus)

# Filter punctuation and lowercase tokens
en_words = filt(en_doc)
en_cor_words = filt(en_cor_doc)
es_words = filt(es_doc)
es_cor_words = filt(es_cor_doc)

# Get frequency of tokens with counter
en_freq = Counter(en_words)
en_cor_freq = Counter(en_cor_words)
es_freq = Counter(es_words)
es_cor_freq = Counter(es_cor_words)

# Convert counter object to dataframe
df_en = dict2df(en_freq)
df_en_cor = dict2df(en_cor_freq)
df_es = dict2df(es_freq)
df_es_cor = dict2df(es_cor_freq)


# Get number of tokens and types
no_tokens_en = len(en_words)
no_tokens_en_cor = len(en_cor_words)
no_tokens_es = len(es_words)
no_tokens_es_cor = len(es_cor_words)
no_types_en = len(en_freq)
no_types_en_cor = len(en_cor_freq)
no_types_es = len(es_freq)
no_types_es_cor = len(es_cor_freq)

# Get Kendall's rank correlation
tau_en = kendcorr(df_en['length'], df_en['frequency'])
tau_en_cor = kendcorr(df_en_cor['length'], df_en_cor['frequency'])
tau_es = kendcorr(df_es['length'], df_es['frequency'])
tau_es_cor = kendcorr(df_es_cor['length'], df_es_cor['frequency'])

# Print datapoints for each corpus
print(f'en poetry: {no_tokens_en} tokens; {no_types_en} types; K tau, p value = {tau_en}')
print(f'en UN corpus: {no_tokens_en_cor} tokens; {no_types_en_cor} types; K tau, p value = {tau_en_cor}')
print(f'es poetry: {no_tokens_es} tokens; {no_types_es} types; K tau, p value = {tau_es}')
print(f'es UN corpus: {no_tokens_es_cor} tokens; {no_types_es_cor} types; K tau, p value = {tau_es_cor}')

# Plot each graph
df_en.plot.scatter(x='length', y='frequency')
plt.xticks(df_en['length'])
plt.title('English poetry')
plt.show()

df_en_cor.plot.scatter(x='length', y='frequency')
plt.xticks(df_en_cor['length'])
plt.title('English UN corpus')
plt.show()

df_es.plot.scatter(x='length', y='frequency')
plt.xticks(df_es['length'])
plt.title('Spanish poetry')
plt.show()

df_es_cor.plot.scatter(x='length', y='frequency')
plt.xticks(df_es_cor['length'])
plt.title('Spanish UN corpus')
plt.show()

# Write dataframes to csv
df_en.to_csv('data/en_poetry.csv')
df_es.to_csv('data/es_poetry.csv')
df_en_cor.to_csv('data/en_corpus.csv')
df_es_cor.to_csv('data/es_corpus.csv')
