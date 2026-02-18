import torch
import requests
import gzip
import json
import random
import os
from transformers import DistilBertTokenizerFast

# URLs from the assignment notebook
GENRE_URL_DICT = {'poetry':                 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
                  'children':               'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
                  'comics_graphic':         'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
                  'fantasy_paranormal':     'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
                  'history_biography':      'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
                  'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
                  'romance':                'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
                  'young_adult':            'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'}

class GoodreadsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_reviews(url, head=5000, sample_size=500):
    """Downloads and samples reviews."""
    reviews = []
    count = 0
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

def prepare_data(model_name='distilbert-base-cased', max_length=512):
    """Loads data, splits it, and tokenizes it."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    
    print("Loading data...")
    for genre, url in GENRE_URL_DICT.items():
        reviews = load_reviews(url)
        # 80-20 Split
        split_idx = int(0.8 * len(reviews))
        train_texts.extend(reviews[:split_idx])
        train_labels.extend([genre] * split_idx)
        test_texts.extend(reviews[split_idx:])
        test_labels.extend([genre] * (len(reviews) - split_idx))

    # Create label maps
    # unique_labels = list(set(train_labels))
    unique_labels = sorted(list(set(train_labels)))
    
    label2id = {label: id for id, label in enumerate(unique_labels)}
    id2label = {id: label for label, id in label2id.items()}

    #label2id = {label: id for id, label in enumerate(unique_labels)}
    #id2label = {id: label for label, id in label2id.items()}

    print("Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    train_dataset = GoodreadsDataset(train_encodings, train_labels_encoded)
    test_dataset = GoodreadsDataset(test_encodings, test_labels_encoded)

    return train_dataset, test_dataset, label2id, id2label