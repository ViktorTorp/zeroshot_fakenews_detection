# Create a script that will use BERT from the transformer library to perform zero-shot classification
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from tqdm import tqdm

def get_model(model_name="facebook/bart-large-mnli"):
    # load model
    model = pipeline("zero-shot-classification", model=model_name)
    return model

def get_prediction(model, statement, labels):
    # get prediction
    prediction = model(statement, candidate_labels=labels)
    return prediction

def get_prediction_labels(prediction):
    # get prediction label
    prediction_labels = prediction['labels']
    return prediction_labels

def get_prediction_scores(prediction):
    # get prediction score
    prediction_scores = prediction['scores']
    return prediction_scores

def get_prediction_label_and_score(prediction):
    # get prediction label and score
    prediction_labels = get_prediction_labels(prediction)
    prediction_scores = get_prediction_scores(prediction)
    # find the highest score and label
    max_idx = np.argmax(prediction_scores)
    prediction_label = prediction_labels[max_idx]
    prediction_score = prediction_scores[max_idx]
    return prediction_label, prediction_score

def get_transformer_prediction(df, text_col, labels, model_name="facebook/bart-large-mnli"):
    # load model
    model = get_model(model_name=model_name)
    # get predictions, # get prediction label and score    
    predictions = []
    scores = []
    for statement in tqdm(df[text_col]):
        prediction = get_prediction(model, statement, labels)
        prediction_label, prediction_score = get_prediction_label_and_score(prediction)
        predictions.append(prediction_label)
        scores.append(prediction_score)
    # add predictions and scores to df
    df['prediction'] = predictions
    df['score'] = scores
    return df
