from helper_functions import *
import logging
import pandas as pd

import streamlit as st

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def filter_items(query):
    """
    Function to retrieve the most relevant items based on the given query.

    Args:
    - query (str): The search term entered by the user.

    Returns:
    - pd.DataFrame: A DataFrame containing the top matching offers and their similarity scores.
    """
    logger.info(f"Query: {query}")
    try:
        # Get top similar items from offers, brands, and categories
        offer_similarities = get_topn_similar(query, offers)
        brand_similarities = get_topn_similar(query, brand_df)
        category_similarities = get_topn_similar(query, categories)
    except Exception as e:
        logger.error(f"Error getting similarities: {e}")
        return pd.DataFrame()
    # Check if the top offer similarity is greater than the top brand and category similarity
    if (
        offer_similarities["similarity"].iloc[0]
        > brand_similarities["similarity"].iloc[0] - 0.1
        and offer_similarities["similarity"].iloc[0]
        > category_similarities["similarity"].iloc[0] - 0.1
    ):
        return offer_similarities[["OFFER", "similarity"]]
    # Check if top brand similarity is greater than top offer similarity
    elif (
        brand_similarities["similarity"].iloc[0] - 0.1
        > offer_similarities["similarity"].iloc[0]
    ):
        # If the top brand similarity is also greater than the top category similarity
        if (
            brand_similarities["similarity"].iloc[0]
            > category_similarities["similarity"].iloc[0]
        ):
            # Get offers within the top matching brand
            offers_in_brand = offers[
                offers["BRAND"] == brand_similarities["BRAND"].iloc[0]
            ]
            ret_offer_similarities = get_topn_similar(query, offers_in_brand)
            if ret_offer_similarities!=None and len(ret_offer_similarities) > 0:
                # Return the top offers in the brand if they exist
                return ret_offer_similarities[["OFFER", "similarity"]]
            else:
                # Otherwise, if the category similarity is close to the offer similarity, return top offers
                if (
                    category_similarities["similarity"].iloc[0] - 0.15
                    < offer_similarities["similarity"].iloc[0]
                ):
                    return offer_similarities[["OFFER", "similarity"]]
        # If the top category similarity is higher, get offers matching that category
        ret_df = offer_similarities[
            offer_similarities["PRODUCT_CATEGORY"]
            == category_similarities["PRODUCT_CATEGORY"].iloc[0]
        ]
        # Return offers in the category if they exist
        if ret_df!=None and len(ret_df) > 0:
            return ret_df[["OFFER", "similarity"]]
        else:
            # Otherwise, check for offers in the parent category
            parent_category = category_similarities["IS_CHILD_CATEGORY_TO"].iloc[0]
            ret_df = offer_similarities[
                offer_similarities["IS_CHILD_CATEGORY_TO"] == parent_category
            ]
            # Return offers in the parent category if they exist
            if ret_df!=None and len(ret_df) > 0:
                return ret_df[["OFFER", "similarity"]]
            else:
                # If no matches are found in either category, return the top offers
                return offer_similarities[["OFFER", "similarity"]]
    # Log when the end of the function is reached without a specific return
    logger.info("Reached end of function.")
    return pd.DataFrame()


def app():
    """
    Main Streamlit app function that takes user input and displays the results.

    """
    try:
        query = st.text_input("Enter search term and press Enter:", value="")
        if not query:
            st.markdown(
                "<style>body{background-color: black;}</style>", unsafe_allow_html=True
            )
            return
        filtered_items = filter_items(query)
        if len(filtered_items) > 0:
            filtered_items = filtered_items.head(n=20)
            st.write("Search results:")
            for i in range(len(filtered_items)):
                st.write(
                    f'{filtered_items.iloc[i]["OFFER"]} ({filtered_items.iloc[i]["similarity"]:.2f})'
                )
        else:
            st.write("No results found.")
    except Exception as e:
        logger.error(f"Error in app function: {e}")
        st.write("An error occurred. Please check logs for details.")


if __name__ == "__main__":
    app()
