import torch
import pytorch_lightning as pl

import os
import requests
import csv

from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Tuple

from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self,
                texts: List[str],
                labels: List[int],
                vocab = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def _tokenize(self, text):
        return self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding="max_length", return_tensors='pt')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tokenized_data = self._tokenize(self.texts[idx])
        return tokenized_data['input_ids'].squeeze(), self.labels[idx]


class AG_News(pl.LightningDataModule):
    def __init__(self,
                data_dir:str,
                batch_size:str,
                out_classes:int,
                vocab_size: int,
                num_workers:int=0):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.out_classes = out_classes
        self.num_workers = num_workers

        self.vocab = None
        self.vocab_size = vocab_size

    def build_vocab(self, folder_path: str) -> dict:
        """
        Build a vocabulary from all .csv files in a given folder. I am using
        frequency-based pruning to cap to N words.
        """

        word_counter = Counter()

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
                    word_counter.update(f.read().split())

        # Build vocabulary from the Counter, keeping only the top `max_words` words
        vocab = {word: idx for idx, (word, _) in enumerate(word_counter.most_common(self.vocab_size-1))}

        # Add <UNK> token to the vocab
        vocab['<UNK>'] = len(vocab)

        return vocab

    def prepare_data(self):
        "Download data from source if it doesn't exist."
        # Class labels:
        classes_txt = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/classes.txt"
        txt_content = requests.get(classes_txt).text
        txt_labs = txt_content.splitlines()
        self.class_labels = [i for i in range(len(txt_labs))]

        # If we want to parse later:
        lab_dict = {}
        for i in range(len(txt_labs)):
            lab_dict[i] = txt_labs[i]
        self.class_label_dict = lab_dict

        # Train and test data files:
        url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"
        folder = self.data_dir + 'AGnews/'
        fname = ['train', 'test']
        for i in range(2):
            filename = folder+"AGnews_"+ fname[i] + ".csv"
            if os.path.exists(filename):
                print('Data already downloaded.')
            else:
                response = requests.get(url + fname[i] + ".csv")
                csv_content = response.text
                clean_content = csv_content.replace("\"", "").replace("\\", "")
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(clean_content)


    def setup(self, stage:str):
        """Build datasets according to Trainer() stage."""
        folder = self.data_dir + 'AGnews/'

        # Build vocabulary
        if self.vocab == None:
            self.vocab = self.build_vocab(folder)

        data = []
        labels = []
        if stage in ['fit', None]:
            with open(folder+"AGnews_train.csv", 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    data.append(' '.join(row[1:]))
                    labels.append(int(row[0])-1)

                train_dataset = TextDataset(texts=data, labels=labels, vocab=self.vocab)

            self.train_dataset, self.val_dataset = random_split(train_dataset, [0.8, 0.2])

        elif stage in ['test', 'predict']:
            with open(folder+"AGnews_test.csv", 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    data.append(' '.join(row[1:]))
                    labels.append(int(row[0])-1)

                self.test_dataset = TextDataset(texts=data, labels=labels, vocab=self.vocab)

                # Prediction same as test, only changes output of function.
                self.pred_dataset = TextDataset(texts=data, labels=labels, vocab=self.vocab)

    def padding(self, batch):
        """Adds padding to the sequences of the same batch so that all have
        the same size. I don't find dynamic padding (grouping batches by size)
        necessary for now.
        """
        texts, labels = zip(*batch)
        texts = pad_sequence(texts, batch_first=True)  # Pads the sequences
        return texts, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = True,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         collate_fn = self.padding)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle = False,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         collate_fn = self.padding)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle = False,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         collate_fn = self.padding)

    def predict_dataloader(self):
        predict_dl = DataLoader(self.test_dataset, shuffle = False,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         collate_fn = self.padding)

        # Compute the true classes:
        true_classes = []
        for _, classes in predict_dl:
            true_classes.append(torch.tensor(classes))
        self.true_classes = torch.cat(true_classes)
        return predict_dl


    def teardown(self, stage:str):
        ''' Used to clean-up when the run is finished.'''
        print('Teardown process underway.')


if __name__ == "__main__":
    _ = TextDataset()
    _ = AG_News()
