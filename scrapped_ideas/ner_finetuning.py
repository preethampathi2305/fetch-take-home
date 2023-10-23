import spacy
from spacy.training import Example
import pandas as pd
import random
from spacy.util import minibatch, compounding

# Load the data
train_df = pd.read_csv("training_data.csv")

# Prepare data
data = []
for index, row in train_df.iterrows():
    if pd.notna(row["Start_Char"]):
        entities = [(int(row["Start_Char"]), int(row["End_Char"]), row["Entity_Label"])]
    else:
        entities = []
    data.append((row["Sentence"], {"entities": entities}))

nlp = spacy.load("en_core_web_sm")

# Split data into training and validation sets
random.shuffle(data)
split_point = int(0.8 * len(data))
train_data = data[:split_point]
valid_data = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in data[split_point:]]

# Get the NER pipe
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add new entity labels to ner
for _, annotations in data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Training configuration
optimizer = nlp.resume_training()
iterations = 10
dropout = 0.5

# Training loop
for itn in range(iterations):
    random.shuffle(train_data)
    losses = {}
    
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        examples = [Example.from_dict(nlp(text), annot) for text, annot in zip(texts, annotations)]
        nlp.update(examples, drop=dropout, losses=losses)
    
    # Evaluation (print F1-score as an example)
    evaluation = nlp.evaluate(valid_data)
    print(f"Iteration #{itn + 1}, Loss: {losses['ner']}")
    for ent_type, scores in evaluation["ents_per_type"].items():
        print(f"Entity {ent_type}, F1: {scores['f']}, Precision: {scores['p']}, Recall: {scores['r']}")

# Save the model
nlp.to_disk("models/brand_ner")
