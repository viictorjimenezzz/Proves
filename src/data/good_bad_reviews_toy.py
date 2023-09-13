import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict

import os
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

        # Tokenization:
        self.tokenized_texts = [self._tokenize(text) for text in texts]

    def _tokenize(self, text):
        return [self.vocab.get(token, 0) for token in text.split()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.tensor(self.tokenized_texts[idx]), self.labels[idx]

class GBReviewsToy(pl.LightningDataModule):
    def __init__(self,
                data_dir:str,
                batch_size:int,
                out_classes:int,
                num_workers:int=0,
                vocab_size=None # not necessary here, but helps with configs
                ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.out_classes = out_classes
        self.num_workers = num_workers

        self.vocab = None # vocabulary

    def build_vocab(self, folder_path: str) -> dict:
        """
        Build a vocabulary from all .txt files in a given folder.
        """
        word_counter = Counter()

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as f:
                    word_counter.update(f.read().split())

        # Build vocabulary from the Counter
        vocab = {word: idx for idx, (word, _) in enumerate(word_counter.most_common())}
        return vocab

    def prepare_data(self):
        """Toy data, no need to download."""

    def setup(self, stage: str):
        '''Build datasets according to Trainer() stage.'''
        folder = self.data_dir + 'good_bad_reviews_toy/'

        # Build vocabulary
        if self.vocab == None:
            self.vocab = self.build_vocab(folder)

        texts = []
        labels = []
        if stage == 'fit' or stage is None:
            with open(folder+'train.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    text, label = line.strip().split("\t")
                    texts.append(text)
                    labels.append(int(label))
                self.train_dataset = TextDataset(texts, labels, self.vocab)

            texts = []
            labels = []
            with open(folder+'validate.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    text, label = line.strip().split("\t")
                    texts.append(text)
                    labels.append(int(label))
                self.val_dataset = TextDataset(texts, labels, self.vocab)

        elif stage == 'test':
            with open(folder+'test.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    text, label = line.strip().split("\t")
                    texts.append(text)
                    labels.append(int(label))
                self.test_dataset = TextDataset(texts, labels, self.vocab)

        elif stage == 'predict':
            with open(folder+'predict.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    text, label = line.strip().split("\t")
                    texts.append(text)
                    labels.append(int(label))
                self.pred_dataset = TextDataset(texts, labels, self.vocab)

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
        predict_dl = DataLoader(self.pred_dataset, shuffle = False,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers,
                         collate_fn = self.padding)


        # Compute the true classes:
        true_classes = []
        for _, classes in predict_dl:
            true_classes.append(classes)

        self.true_classes = [item for sublist in true_classes for item in sublist]
        return predict_dl

    def teardown(self, stage:str):
        ''' Used to clean-up when the run is finished.'''
        print('Teardown process underway.')


if __name__ == "__main__":
    _ = TextDataset()
    _ = GBReviewsToy()
