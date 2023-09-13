import torch
import pytorch_lightning as pl
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import requests
import zipfile
from pathlib import Path

class pizza_steak_sushi(pl.LightningDataModule):
    def __init__(self,
                data_dir:str,
                batch_size:int,
                out_classes:int,
                num_workers:int = 0
                ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.out_classes = out_classes
        self.num_workers = num_workers

        # I write it here because it is the same for all:
        self.transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor()])

    def prepare_data(self):
        '''Download the data. Criterion: if the folders exist, the data is
        assumed to be there.
        '''
        data_path = Path(self.data_dir)
        self.image_path = data_path / "pizza_steak_sushi"

        # If the image folder doesn't exist, download it and prepare it...
        if self.image_path.is_dir():
            print(f"{self.image_path} directory exists, data assumed to be there.")
            print("If the data is not there, delete folders and try again.")
        else:
            print(f"Did not find {self.image_path} directory, creating one...")
            self.image_path.mkdir(parents=True, exist_ok=True)

            # Download pizza, steak, sushi data
            with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
                request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
                print("Downloading pizza, steak, sushi data...")
                f.write(request.content)

            # Unzip pizza, steak, sushi data
            with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
                print("Unzipping pizza, steak, sushi data...")
                zip_ref.extractall(self.image_path)

            # Remove zip file
            os.remove(data_path / "pizza_steak_sushi.zip")

    def setup(self, stage: str):
        '''Build datasets according to Trainer() stage.'''
        if stage == 'fit':
            full_train_data = datasets.ImageFolder(self.image_path / 'train', transform=self.transform)
            self.train_data, self.val_data = random_split(full_train_data, [0.8, 0.2])

        if stage in ['test', 'predict']:
            self.test_data = datasets.ImageFolder(self.image_path / 'test', transform=self.transform)



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
    _ = pizza_steak_sushi()
