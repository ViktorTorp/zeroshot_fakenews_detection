# make script which can calculate the number of gpt2 tokens, using huggingsface tokenizer, in a culumn of a dataframe
# and then estimate the price of the tokens

import pandas as pd
from transformers import GPT2TokenizerFast


def get_tokenizer():
    # load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return tokenizer

def get_token_count(text, tokenizer):
    # get token count
    tokens = tokenizer.encode(text, return_tensors='pt')
    return len(tokens[0])

def get_price(token_count, token_price=0.00000):
    # get price
    price = token_count * token_price
    return price

def get_price_estimate(df, text_col, tokenizer, token_price=0.00000):
    # get price estimate
    df['token_count'] = df[text_col].apply(lambda x: get_token_count(x, tokenizer))
    df['price'] = df['token_count'].apply(lambda x: get_price(x, token_price=token_price))
    return df

def calc_price(df, text_col='statement', token_price=0.00000):
    # load tokenizer
    tokenizer = get_tokenizer()
    # get price estimate
    df = get_price_estimate(df, text_col, tokenizer, token_price=token_price)
    # Calculate total price
    total_price = df['price'].sum()
    return token_price


def read_data(path_to_data):
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state_info', 'party', 'barely_true_count', 'false_count', 'half_true_count', 'mostly_true_count', 'pants_on_fire_count', 'context']
    # read data
    df = pd.read_csv(path_to_data, sep="\t", names=columns)
    return df


if __name__ == '__main__':
    # Text column
    text_col = 'statement'
    # Token price
    token_price = 0.0200/1000

    df = read_data('data/liar_dataset/train.tsv')
    train_price = calc_price(df, text_col=text_col, token_price=token_price)
    df = read_data('data/liar_dataset/valid.tsv')
    val_price = calc_price(df, text_col=text_col, token_price=token_price)
    df = read_data('data/liar_dataset/test.tsv')
    test_price = calc_price(df, text_col=text_col, token_price=token_price) 
    print('Total price: ', train_price + val_price + test_price)
