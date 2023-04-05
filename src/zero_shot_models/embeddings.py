import os
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
import spacy


def get_spacy_embeddings(text, nlp):
    # get embeddings
    doc = nlp(text)
    return doc.vector

def get_cosine_similarity(embeddings1, embeddings2):
    # get cosine similarity
    return cosine_similarity([embeddings1], [embeddings2])[0][0]

def get_spacy_predictions(df, text_col, labels):
    # load spacy model
    nlp = spacy.load("en_core_web_lg")
    # get columns
    cols = list(df.columns)
    # get label embeddings
    label_embeddings = [get_spacy_embeddings(label, nlp) for label in labels]
    # get text embeddings
    df['embeddings'] = df[text_col].apply(lambda x: get_spacy_embeddings(x, nlp))
    # get cosine similarity scores
    for label, emb in zip(labels, label_embeddings):
        df[label] = df['embeddings'].apply(lambda x: get_cosine_similarity(x, emb))

    # get max score
    df['max_score'] = df[labels].max(axis=1)
    # get label with max score
    df['prediction'] = df[labels].idxmax(axis=1)
    return df[["prediction", "max_score"] + cols]
