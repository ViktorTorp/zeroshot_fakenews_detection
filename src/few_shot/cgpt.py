import pandas as pd
import openai
import os
import json
import time
import numpy as np
from transformers import GPT2TokenizerFast
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

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

# def get_prompt(statement, binary=False):
#    if binary:
#        labels = "'false', or 'true'"
#    else:
#        labels = "'false', 'half-true', 'mostly-true', 'true', 'pants on fire', 'barely-true', or 'pants-fire'"

    # include the subject, speaker, speaker_job, state_info, party and ceontext in the prompt
#    statement_intro = f"""Classify the validity of the following statements as either {labels}
#Hillary Clinton in 2005 co-sponsored legislation that would jail flag burners. => true
#Only 2 percent of public high schools in the country offer PE classes. => false\n"""
#    prompt = f"""{statement_intro}{statement} =>"""
#    return prompt

def get_prompt(statement, binary=False):
    if binary:
        labels = "'false', or 'true'"
    else:
        labels = "'false', 'half-true', 'mostly-true', 'true', 'pants on fire', 'barely-true', or 'pants-fire'"

    # include the subject, speaker, speaker_job, state_info, party and ceontext in the prompt
    statement_intro = f"""Classify the validity of the following statements as either {labels}
Under Mayor Cicilline, [Providence] was a sanctuary city. => false
Hillary Clinton in 2005 co-sponsored legislation that would jail flag burners. => true
Only 2 percent of public high schools in the country offer PE classes. => false
Says U.S. Rep. Steve Southerland voted to keep the shutdown going. => true\n"""
    prompt = f"""{statement_intro}{statement} =>"""
    return prompt

def get_response(prompt, model="text-davinci-003"):
    openai.api_key = "SOME_KEY"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.4,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.35,
        presence_penalty=0.75
    )
    return response

def get_label(response):
    try:
        label = response['choices'][0]['text'].strip().lower()
    except:
        label = ''
    return label

def is_label_valid(label):
    valid_labels = ['false', 'half-true', 'mostly-true', 'true', 'pants on fire', 'barely-true', 'pants-fire']
    if label in valid_labels:
        return True
    else:
        return False


def get_label_from_statement(statement, binary=False):
    prompt = get_prompt(statement, binary=binary)
    response = get_response(prompt)
    label = get_label(response)
    return label

def get_label_from_df(df, text_col='statement', binary=False):
    df['prediction'] = df[text_col].apply(lambda x: get_label_from_statement(x, binary=binary))
    return df

def save_results(df, path_to_save):
    df.to_csv(path_to_save, index=False)

def main(text_col='statement', binary=False):
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
    df = get_label_from_df(df, text_col='statement', binary=binary)
    # save results
    save_results(df, path_to_save)

if __name__ == '__main__':
    main()
