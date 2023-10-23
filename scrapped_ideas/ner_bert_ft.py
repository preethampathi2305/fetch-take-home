from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch

class EntityRecognizer:
    def __init__(self):
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("elastic/distilbert-base-cased-finetuned-conll03-english")
        self.model = DistilBertForTokenClassification.from_pretrained("elastic/distilbert-base-cased-finetuned-conll03-english")
        self.model.eval()

        # Define the label set for the CoNLL03 task
        self.labels = ["O",       # Outside of a named entity
                       "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
                       "I-MISC",  # Miscellaneous entity
                       "B-PER",  # Beginning of a person's name right after another person's name
                       "I-PER",  # Person's name
                       "B-ORG",  # Beginning of an organization right after another organization
                       "I-ORG",  # Organization
                       "B-LOC",  # Beginning of a location right after another location
                       "I-LOC"]  # Location

    def predict_entities(self, text):
        encoding = self.tokenizer.encode_plus(text, return_tensors="pt", return_attention_mask=True, return_offsets_mapping=True)
        inputs = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offsets = encoding["offset_mapping"]

        with torch.no_grad():
            outputs = self.model(inputs, attention_mask=attention_mask).logits
            predictions = torch.argmax(outputs, dim=2)[0]

        entities = []
        entity_tokens = []
        entity_type = None
        for idx, (token, label_id) in enumerate(zip(self.tokenizer.tokenize(text), predictions)):
            label = self.labels[label_id]
            
            start, end = offsets[0][idx]
            word = text[start:end]
            
            if label == "O":
                if entity_tokens:
                    entities.append((" ".join(entity_tokens), entity_type))
                    entity_tokens = []
                    entity_type = None
            else:
                entity_class, entity_label = label.split('-')
                if entity_class == "B":
                    if entity_tokens:
                        entities.append((" ".join(entity_tokens), entity_type))
                        entity_tokens = []
                    entity_tokens.append(word)
                    entity_type = entity_label
                elif entity_class == "I":
                    entity_tokens.append(word)
                    
        if entity_tokens:
            entities.append((" ".join(entity_tokens), entity_type))

        return entities