import torch
from torch import nn
from torchvision.models import resnet50

class Resnet50(nn.Module):
    def __init__(self, out_features:int) -> None :
        super().__init__()

        pretrained_model = resnet50(weights = 'DEFAULT')
        num_filters = pretrained_model.fc.in_features # get output shape of feat. extr.
        num_layers = list(pretrained_model.children())[:-1] # extract model
        self.feature_extractor = nn.Sequential(*num_layers) # build from pretrained
        self.feature_extractor.requires_grad_(False) # freeze Resnet50
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, out_features),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Eval mode in feature_extractor, since we are not training it

        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        # Training only involves this part:
        return self.classifier(representations)

if __name__ == "__main__":
    _ = Resnet50()
