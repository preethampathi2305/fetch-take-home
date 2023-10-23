import random
import pandas as pd
import nltk
from nltk.corpus import brown

brands = pd.read_csv('brand_category.csv')['BRAND'].astype(str).str.lower().tolist()
retailers = pd.read_csv('offer_retailer.csv')['RETAILER'].astype(str).str.lower().tolist()
# categories = pd.read_csv('categories.csv')['PRODUCT_CATEGORY'].astype(str).str.lower().tolist()

# Modified and new contexts for brands
brand_contexts = [
    "{} is one of my favorite brands",
    "Have you tried the new {} product",
    "I think {} makes high-quality items",
    "My friend works at {}",
    "I just bought a new gadget from {}",
    "i heard {} is having a sale next week",
    "You should check out {}'s latest offerings",
    "The quality of {} products is unmatched",
    "I saw an advertisement for {} on TV",
    "We get our supplies from {}",
    "Do you have products from {}",
    "the company {} was recommended to me by a friend",
    "I've been using products from {} for years",
    "There's a {} store near my house",
    "I read a review about {} online",
    "Some random product at {}",
    ""
]

# Contexts for categories
# category_contexts = [
#     "Have you checked the latest in {}",
#     "The {} section has some good choices",
#     "I am thinking of exploring more in the {} category",
#     "They have a diverse range of products under {}",
#     "I found some unique items in the {} aisle",
#     "If you're interested in {}, they have a great selection",
#     "There's a sale in the {} section today",
#     "I'm looking for recommendations in {}",
#     "What's your favorite product from the {} category",
#     "I've always been fascinated by {} products"
# ]

brand_contexts = [context.lower() for context in brand_contexts]
# category_contexts = [context.lower() for context in category_contexts]

# Extract words from the Brown corpus
brown_words = list(brown.words())
brown_words = [word.lower() for word in brown_words]
brown_words = [word for word in brown_words if word.isalpha()]

# Function to generate a random sentence using words from the Brown corpus
def generate_random_sentence(length=8):
    return ' '.join(random.choices(brown_words, k=length))

# Generate a list of random sentences
num_random_sentences = 2000
random_sentences = [generate_random_sentence() for _ in range(num_random_sentences)]

# Convert all existing sentences and entities to lowercase
brand_sentences = [random.choice(brand_contexts).format(brand.lower()) for brand in brands]
retailer_sentences = [random.choice(brand_contexts).format(retailer.lower()) for retailer in retailers]
# category_sentences = [random.choice(category_contexts).format(category.lower()) for category in categories]

# Generate the output format
output_data = []

# For brands
for i in range(len(brands)):
    brand_name = brands[i]
    brand_sentence = brand_sentences[i]
    start_char = brand_sentence.index(brand_name)
    end_char = start_char + len(brand_name)
    output_data.append([brand_sentence, start_char, end_char, "ORG"])

# For retailers
for i in range(len(retailers)):
    retailer_name = retailers[i]
    retailer_sentence = retailer_sentences[i]
    start_char = retailer_sentence.index(retailer_name)
    end_char = start_char + len(retailer_name)
    output_data.append([retailer_sentence, start_char, end_char, "ORG"])

# For categories
# for i in range(len(categories)):
#     category_name = categories[i]
#     category_sentence = category_sentences[i]
#     start_char = category_sentence.index(category_name)
#     end_char = start_char + len(category_name)
#     output_data.append([category_sentence, start_char, end_char, "CAT"])

# Adding random sentences without entities
for sentence in random_sentences:
    output_data.append([sentence, None, None, None])

# Shuffle the data for randomness
random.shuffle(output_data)

output_df = pd.DataFrame(output_data, columns=["Sentence", "Start_Char", "End_Char", "Entity_Label"])
output_df.to_csv("training_data.csv", index=False)