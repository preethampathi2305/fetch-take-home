
# Documentation for Search App

## Overview
The Search App provides an interface for users to input a query and get the top search results based on the query. It leverages embeddings to rank and display the most relevant items.

## Files

### 1. search.py

#### Imports:
- Functions from `helper_functions.py`
- `streamlit` for app interface
- `pandas` for data manipulation
- `torch` for PyTorch operations

#### Data Files:
- brand_embeddings.csv: Contains embeddings for brands
- categories_embeddings.csv: Contains embeddings for categories
- offers_embeddings.csv: Contains embeddings for offers

#### Functions:

- **filter_items(query)**: Filters items based on their similarities to a given query. It prioritizes items by checking similarity scores among offers, brands, and categories.

- **app()**: The main function for the Streamlit application. It provides an input interface for users to enter a query and displays the top search results based on the query.

### 2. helper_functions.py

#### Libraries:
- `ast`: For converting strings to lists
- `pandas`: For data manipulation
- `torch`: For PyTorch operations
- `transformers`: For tokenization and embeddings

#### Embedding Model:
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings

#### Functions:

- **str_to_list(s)**: Converts a string to a list.
- **mean_pooling(model_output, attention_mask)**: Computes mean pooling for embeddings.
- **get_embeddings(df, text_col, model, tokenizer)**: Returns embeddings for a given dataframe column.
- **get_topn_similar(text, df)**: Returns top N similar items based on text similarity.

---

*Note: For detailed functionality, refer to the source code.*


## How to Run the App

1. Ensure you have all the required libraries installed. You can install them using `pip`:
```
pip install streamlit pandas torch transformers
```

2. Navigate to the directory containing the `search.py` script.

3. Run the app using the following command:
```
streamlit run search.py
```

4. The app should open in your default web browser. If it doesn't, you'll see a URL in the terminal (usually `http://localhost:8501`). Copy and paste that URL into your browser to access the app.

5. Enter your search query in the input box and view the results.

---

