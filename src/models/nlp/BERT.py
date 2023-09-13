import torch
from torch import nn

from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, out_classes:int):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.requires_grad_(False) # transformers model (HF)
        self.fc = nn.Linear(768, out_classes)  # 768 is the size of BERT's hidden state

    def forward(self, x):
        out = self.bert(x).last_hidden_state
        out = self.fc(out[:, 0, :])  # using the CLS token representation for classification
        return out.softmax(dim=1)  # out


    if __name__ == "__main__":
        _ = BertClassifier()
