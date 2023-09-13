import torch
from torch import nn

class LSTMClassifier(nn.Module):
    def __init__(self,
                vocab_size:int,
                embedding_dim:int,
                hidden_size:int,
                num_layers:int,
                dropout:float,
                out_classes:int):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            dropout=dropout,
                            batch_first=True) # expects batch_size at position 0
        self.fc = nn.Linear(hidden_size, out_classes)

    def forward(self, x):
        print('\n\n\n')
        print(x)
        print('\n\n\n')
        out = self.embedding(x)

        out, _ = self.lstm(out) # lstm expects (batch_size, sequence_length, input_size)
        out = self.fc(out[:, -1, :])
        return out.softmax(dim=1) # out


    if __name__ == "__main__":
        _ = LSTMClassifier()
