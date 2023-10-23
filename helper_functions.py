import ast 
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def str_to_list(s):
    return ast.literal_eval(s)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(df, text_col, model, tokenizer):
    model.eval()
    encoded_input = tokenizer(list(df[text_col]), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return df.assign(embeddings=sentence_embeddings.tolist())

def get_topn_similar(text, df):
    text_embedding = get_embeddings(pd.DataFrame({'OFFER': [text]}), 'OFFER', embedding_model, embedding_tokenizer)['embeddings'].values[0]
    df['similarity'] = df['embeddings'].apply(lambda x: np.dot(x, text_embedding))
    return df.sort_values(by='similarity', ascending=False)