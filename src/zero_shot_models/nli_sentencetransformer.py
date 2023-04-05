# create a script that will use cross-encoder from the transformer library to perform zero-shot classification using NLI
import pandas as pd
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm


def get_model(model_name="cross-encoder/nli-distilroberta-base"):
    # load model
    model = CrossEncoder(model_name)
    return model

def preb_statement_labels(statement, labels):
    label_entailments = [f"the accuracy of this statement is {label}" for label in labels]
    statement_label_tuples = [(statement, label_entailment) for label_entailment in label_entailments]
    return statement_label_tuples

def _get_prediction(model, statement, labels):
    # get prediction
    statement_label_tuples = preb_statement_labels(statement, labels)
    prediction = model.predict(statement_label_tuples)
    return prediction

def get_prediction_label_score(prediction, labels):
    # Get entailment scores
    entailment_scores = prediction[:, 1]
    highest_entailment_score = np.argmax(entailment_scores)
    label = labels[highest_entailment_score]
    return label, entailment_scores


def get_nli_transformer_prediction(df, text_col, labels, model_name="cross-encoder/nli-distilroberta-base"):
    # load model
    model = get_model(model_name=model_name)
    # get predictions, # get prediction label and score    
    predictions = []
    scores = []
    for statement in tqdm(df[text_col]):
        prediction = _get_prediction(model, statement, labels)
        prediction_label, prediction_score = get_prediction_label_score(prediction, labels)
        predictions.append(prediction_label)
        scores.append(prediction_score)
    # add predictions and scores to df
    df['prediction'] = predictions
    df['score'] = scores
    return df
