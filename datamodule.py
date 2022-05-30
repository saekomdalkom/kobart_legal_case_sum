# extends Lightning Data Module
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_file,
                 test_file, 
                 tokenizer,
                 max_len=512,
                 batch_size=10,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.trainDataset = MyDataset(self.train_file_path,
                                 self.tokenizer,
                                 self.max_len)
        self.testDataset = MyDataset(self.test_file_path,
                                self.tokenizer,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.trainDataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, 
                           shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.testDataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, 
                         shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.testDataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=False)
        return test



class MyDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, pad_index = 0, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['decision'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['issue'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}
    
    def __len__(self):
        return self.len
