import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class caltech101(pl.LightningDataModule):
    def __init__(self,
                data_dir:str,
                batch_size:int,
                out_classes:int,
                num_workers:int=0,
                ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.out_classes = out_classes
        self.num_workers = num_workers

        # I write it here because it is the same for all:
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        # Some pics were not (3,224,224)
                        transforms.Lambda(lambda img : img.convert('RGB')),
                        transforms.ToTensor()])

    def prepare_data(self):
        self.dataset = datasets.Caltech101(self.data_dir,
                                            download=True,
                                            transform=self.transform)

    def setup(self, stage: str):
        '''Build datasets according to Trainer() stage.'''
        train_dataset, test_dataset = random_split(self.dataset, [0.8, 0.2])

        if stage == 'fit':
            self.train_data, self.val_data = random_split(train_dataset, [0.8, 0.2])

        if stage in ['test', 'predict']:
            self.test_data = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle = True,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle = False,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle = False,
                         batch_size = self.batch_size,
                         num_workers = self.num_workers)

    def predict_dataloader(self):
        # Same as test by now. I will keep it in case I wanna access it later.
        self.pred_dl = DataLoader(self.test_data, shuffle = False,
                                  batch_size = self.batch_size,
                                  num_workers = self.num_workers)


        # Compute the true classes:
        true_classes = []
        for _, classes in self.pred_dl:
            true_classes.append(classes)
        self.true_classes = torch.cat(true_classes) # as attribute

        return self.pred_dl # only return dataloader

    def teardown(self, stage:str):
        ''' Used to clean-up when the run is finished.'''
        print('Teardown process underway.')



if __name__ == "__main__":
    _ = caltech101()
