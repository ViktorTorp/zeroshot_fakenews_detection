import pandas as pd
import openai
import os
import json
import time
import numpy as np
from transformers import GPT2TokenizerFast
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from tqdm import tqdm

def get_tokenizer():
    # load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return tokenizer

def get_token_count(text, tokenizer):
    # get token count
    tokens = tokenizer.encode(text, return_tensors='pt')
    return len(tokens[0])

def get_price(token_count, token_price=0.02000):
    # get price
    price = token_count * token_price
    return price

def get_price_estimate(df, text_col, tokenizer, token_price=0.02000):
    # get price estimate
    df['token_count'] = df[text_col].apply(lambda x: get_token_count(x, tokenizer))
    df['price'] = df['token_count'].apply(lambda x: get_price(x, token_price=token_price))
    return df

def calc_price(df, prompt, text_col='statement', token_price=0.02000):
    # load tokenizer
    tokenizer = get_tokenizer()
    # get price estimate
    df = get_price_estimate(df, text_col, tokenizer, token_price=token_price)
    # Calculate total price
    total_price = df['price'].sum()
    # Add price of prompt
    total_price += get_price(get_token_count(prompt, tokenizer), token_price=token_price) * len(df)
    return total_price

def read_data(path_to_data):
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info', 'party', 'barely_true_count', 'false_count', 'half_true_count', 'mostly_true_count', 'pants_on_fire_count', 'context']
    # read data
    df = pd.read_csv(path_to_data, sep="\t", names=columns)
    return df

def get_prompt(statement, binary=False, include_speaker=False, include_subject=False, include_job_title=False, include_context=False):
    if binary:
        labels = "'false', or 'true'"
    else:
        labels = "'false', 'half-true', 'mostly-true', 'true', 'pants on fire', 'barely-true', or 'pants-fire'"
    # include_speaker is a string
    if type(include_speaker) == str:
        statement_intro = f"""Classify the accuracy of the following statement as either {labels}.\n"""

        statement_info = f"""Statement: "{statement}", subject: {include_subject}, speaker: {include_speaker}, job title: {include_job_title}, context: {include_context}.\n"""
        response_format = f"You response should be of the following structure:[label]: [explanation]"""
        return statement_intro + statement_info + response_format
    else:
        # include the subject, speaker, speaker_job, state_info, party and ceontext in the prompt
        statement_intro = f"""Classify the accuracy of the following statement: '{statement}'"""
        prompt = f"""{statement_intro}\nYou can only answer with one of the following categories to describe the accuracy of the claim: {labels}. You response should be of the following structure:[label]: [explanation]"""
        return prompt

def get_response(prompt, model="text-davinci-003"):
    openai.api_key = "SOME_API_KEY"
    try:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=0.4,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.35,
            presence_penalty=0.75
        )
    except:
        response = ""
    return response

def get_label(response):
    try:
        label = response['choices'][0]['text'].split(':')[0].strip().lower()
    except:
        label = ""
    return label

def get_explanation(response):
    try:
        explanation = response['choices'][0]['text'].split(':')[1]
        explanation = explanation if explanation[0] != " " else explanation[1:]
    except:
        explanation = ""
    return explanation


def is_label_valid(label):
    valid_labels = ['false', 'half-true', 'mostly-true', 'true', 'pants on fire', 'barely-true', 'pants-fire', '']
    if label in valid_labels:
        return True
    else:
        return False

def get_label_and_explanation(response):
    label = get_label(response)
    valid_label = is_label_valid(label)
    label = label if valid_label else "abstain"
    explanation = get_explanation(response)
    return label, explanation

def get_label_and_explanation_from_statement(statement, binary=False, include_speaker=False, include_subject=False, include_job_title=False, include_context=False):
    prompt = get_prompt(statement, binary=binary, include_speaker=include_speaker, include_subject=include_subject, include_job_title=include_job_title, include_context=include_context)
    response = get_response(prompt)
    label, explanation = get_label_and_explanation(response)
    return label, explanation

def get_label_and_explanation_from_df(df ,text_col='statement', binary=False, include_speaker=False, include_subject=False, include_job_title=False, include_context=False, path_to_save=None):
    # get label and explanation in batches and save to df in batches
    batch_size = 10
    df["prediction"] = [""] * len(df)
    df["explanation"] = [""] *  len(df)
    for i in tqdm(range(0, len(df), batch_size)):
        # get labels and explanations
        labels = []
        explanations = []
        for j in range(i, i+batch_size):
            if j < len(df):
                statement = df[text_col].iloc[j]
                if not include_speaker:
                    label, explanation = get_label_and_explanation_from_statement(statement, binary=binary, include_speaker=include_speaker, include_subject=include_subject, include_job_title=include_job_title, include_context=include_context)
                else:
                    label, explanation = get_label_and_explanation_from_statement(statement, binary=binary, include_speaker=df['speaker'].iloc[j], include_subject=df['subject'].iloc[j], include_job_title=df['speaker_job'].iloc[j], include_context=df['context'].iloc[j])
                # set df at index j to label and explanation
                df.loc[j, 'prediction'] = label
                df.loc[j, 'explanation'] = explanation
            else:
                break
        # save dfs
        if path_to_save is not None:
            df.to_csv(path_to_save)
    return df


def save_results(df, path_to_save):
    df.to_csv(path_to_save, index=False)

def main(text_col='statement', binary=False, include_speaker=False, include_subject=False, include_job_title=False, include_context=False):
    # set path to data
    path_to_data = 'data/liar_dataset/train.tsv'
    # set path to save results
    path_to_save = 'data/liar_dataset/train_with_labels.tsv'
    # read data
    df = read_data(path_to_data)
    # calculate price
    total_price = calc_price(df, get_prompt(''), token_price=0.02000/1000)
    print(f'Total price: {total_price}')
    # get label and explanation
    df = get_label_and_explanation_from_df(df, text_col='statement', binary=binary, include_speaker=include_speaker, include_subject=include_subject, include_job_title=include_job_title, include_context=include_context)
    # save results
    save_results(df, path_to_save)

if __name__ == '__main__':
    main()
