from helper_functions import *
import logging
import pandas as pd

import streamlit as st

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# set torch device to cpu
device = torch.device('cpu')



brand_df = pd.read_csv('data/brand_embeddings.csv')
brand_df['embeddings'] = brand_df['embeddings'].apply(str_to_list)

offers = pd.read_csv('data/offers_embeddings.csv')
offers['embeddings'] = offers['embeddings'].apply(str_to_list)

categories = pd.read_csv('data/categories_embeddings.csv')
categories['embeddings'] = categories['embeddings'].apply(str_to_list)

def get_embeddings(df, text_col, model, tokenizer):
    logger.info(f"Getting embeddings for {len(df)} rows.")
    model.eval()
    logger.info(f"Encoding input...")
    encoded_input = tokenizer(list(df[text_col]), padding=True, truncation=True, return_tensors='pt')
    logger.info(f"Encoded input shape: {encoded_input['input_ids'].shape}")
    with torch.no_grad():
        model_output = model(**encoded_input)
    logger.info(f"Model output shape: {model_output[0].shape}")
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    logger.info(f"Sentence embeddings shape: {sentence_embeddings.shape}")
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    logger.info(f"Embeddings shape: {sentence_embeddings.shape}")
    return df.assign(embeddings=sentence_embeddings.tolist())

def get_topn_similar(text, df):
    text_embedding = get_embeddings(pd.DataFrame({'OFFER': [text]}), 'OFFER', embedding_model, embedding_tokenizer)['embeddings'].values[0]
    df['similarity'] = df['embeddings'].apply(lambda x: np.dot(x, text_embedding))
    return df.sort_values(by='similarity', ascending=False)


def filter_items(query):
    logger.info(f"Query: {query}")
    try:
        offer_similarities = get_topn_similar(query, offers)
        brand_similarities = get_topn_similar(query, brand_df)
        category_similarities = get_topn_similar(query, categories)
        logger.info(f"Offer similarities: {offer_similarities.head(n=5)}")
        logger.info(f"Brand similarities: {brand_similarities.head(n=5)}")
        logger.info(f"Category similarities: {category_similarities.head(n=5)}")
    except Exception as e:
        logger.error(f"Error getting similarities: {e}")
        return pd.DataFrame()
    if offer_similarities['similarity'].iloc[0] > brand_similarities['similarity'].iloc[0]-0.1 and offer_similarities['similarity'].iloc[0] > category_similarities['similarity'].iloc[0]-0.1:
        return offer_similarities[['OFFER', 'similarity']]
    elif brand_similarities['similarity'].iloc[0]-0.1 > offer_similarities['similarity'].iloc[0]:
        if brand_similarities['similarity'].iloc[0] > category_similarities['similarity'].iloc[0]:
            offers_in_brand = offers[offers['BRAND']==brand_similarities['BRAND'].iloc[0]]
            ret_offer_similarities = get_topn_similar(query, offers_in_brand)
            if len(ret_offer_similarities)>0:
                return ret_offer_similarities[['OFFER', 'similarity']]
            else:
                if category_similarities['similarity'].iloc[0]-0.15 < offer_similarities['similarity'].iloc[0]:
                    return offer_similarities[['OFFER', 'similarity']]                    
        ret_df = offer_similarities[offer_similarities['PRODUCT_CATEGORY']==category_similarities['PRODUCT_CATEGORY'].iloc[0]]
        if len(ret_df)>0:
            return ret_df[['OFFER', 'similarity']]
        else:
            parent_category = category_similarities['IS_CHILD_CATEGORY_TO'].iloc[0]
            ret_df = offer_similarities[offer_similarities['IS_CHILD_CATEGORY_TO']==parent_category]
            if len(ret_df)>0:
                return ret_df[['OFFER', 'similarity']]
            else:
                return offer_similarities[['OFFER', 'similarity']]
    logger.info('Reached end of function.')

def app():
    try:
        query = st.text_input('Enter search term and press Enter:', value='')

        if not query:
            st.markdown('<style>body{background-color: black;}</style>', unsafe_allow_html=True)
            return

        filtered_items = filter_items(query)

        if len(filtered_items)>0:
            filtered_items = filtered_items.head(n=20)
            st.write('Search results:')
            for i in range(len(filtered_items)):
                st.write(f'{filtered_items.iloc[i]["OFFER"]} ({filtered_items.iloc[i]["similarity"]:.2f})')
        else:
            st.write('No results found.')
    except Exception as e:
        logger.error(f"Error in app function: {e}")
        st.write("An error occurred. Please check logs for details.")


if __name__ == '__main__':
    app()