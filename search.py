from helper_functions import *

import streamlit as st
import pandas as pd

import torch

brand_df = pd.read_csv('data/brand_embeddings.csv', converters={'embeddings': str_to_list})
categories = pd.read_csv('data/categories_embeddings.csv', converters={'embeddings': str_to_list})
offers = pd.read_csv('data/offers_embeddings.csv', converters={'embeddings': str_to_list})

def filter_items(query):
    offer_similarities = get_topn_similar(query, offers)
    brand_similarities = get_topn_similar(query, brand_df)
    category_similarities = get_topn_similar(query, categories)
    print({
        'offer': offer_similarities['similarity'].iloc[0],
        'brand': brand_similarities['similarity'].iloc[0],
        'category': category_similarities['similarity'].iloc[0]
    })
    if offer_similarities['similarity'].iloc[0] > brand_similarities['similarity'].iloc[0]-0.1 and offer_similarities['similarity'].iloc[0] > category_similarities['similarity'].iloc[0]-0.1:
        print('choice:0')
        return offer_similarities[['OFFER', 'similarity']]
    elif brand_similarities['similarity'].iloc[0]-0.1 > offer_similarities['similarity'].iloc[0]:
        if brand_similarities['similarity'].iloc[0] > category_similarities['similarity'].iloc[0]:
            offers_in_brand = offers[offers['BRAND']==brand_similarities['BRAND'].iloc[0]]
            ret_offer_similarities = get_topn_similar(query, offers_in_brand)
            if len(ret_offer_similarities)>0:
                print('choice:1')
                return ret_offer_similarities[['OFFER', 'similarity']]
            else:
                if category_similarities['similarity'].iloc[0]-0.15 < offer_similarities['similarity'].iloc[0]:
                    print('choice:2')
                    return offer_similarities[['OFFER', 'similarity']]                    
        ret_df = offer_similarities[offer_similarities['PRODUCT_CATEGORY']==category_similarities['PRODUCT_CATEGORY'].iloc[0]]
        if len(ret_df)>0:
            print('choice:3')
            return ret_df[['OFFER', 'similarity']]
        else:
            parent_category = category_similarities['IS_CHILD_CATEGORY_TO'].iloc[0]
            ret_df = offer_similarities[offer_similarities['IS_CHILD_CATEGORY_TO']==parent_category]
            if len(ret_df)>0:
                print('choice:4')
                return ret_df[['OFFER', 'similarity']]
            else:
                print('choice:5')
                return offer_similarities[['OFFER', 'similarity']]



def app():
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


if __name__ == '__main__':
    app()