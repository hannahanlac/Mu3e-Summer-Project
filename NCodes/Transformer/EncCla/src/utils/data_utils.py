import torch
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import logging


def load_dataloader(config, device, mode="all"):
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["dataloader_num_workers"]

    loaders = {}

    if mode == "train" or mode == "all":
        shuffle = config["training"]["shuffle"]
        logging.info("Loading train data with DataLoader")
        train_dataset = ForwardPassDataset(data_dir, config["data"]["train_file"])
        val_dataset = ForwardPassDataset(data_dir, config["data"]["val_file"])
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    if mode == "eval" or mode == "all":
        test_dataset = ForwardPassDataset(data_dir, config["data"]["test_file"])
        test_helper_dataset = ScoringHelperDataset(
            data_dir, config["data"]["test_helperfile"]
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        loaders["test_helper"] = DataLoader(
            test_helper_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn_test,
        )

    return loaders


def load_truths(config):
    data_dir = config["data"]["data_dir"]
    test_truth_filename = config["data"]["test_truthfile"]
    test_truth_filepath = os.path.join(data_dir, test_truth_filename)
    return pd.read_csv(test_truth_filepath)


# dynamically padding data based on the max seq_length of the batch
def collate_fn(batch):
    dat1, dat2 = zip(*batch)
    dat1_padded = pad_sequence(dat1, batch_first=True, padding_value=0.0)
    dat2_padded = pad_sequence(dat2, batch_first=True, padding_value=-1) #As 0 is technically a valid label for us NOTE SHOULD CHANGE
    return dat1_padded, dat2_padded

def collate_fn_test(batch):
    dat1, dat2 = zip(*batch)
    dat1_padded = pad_sequence(dat1, batch_first=True, padding_value=0.0) # Pad the hitIndex values up to the max of the batch: again ensuring consistent size based on batches
    dat2_padded = pad_sequence(dat2, batch_first=True, padding_value= 0.0) # Pad the frame number up to the maximum value. All padded frames of 0 will be for hits of 0... so should avoid issues?
    return dat1_padded, dat2_padded



# def helper_collate_fn(batch):
#     """
#     Collate function for helper dataset.
#     Each sample is a tuple (hit_ids, event_ids).
#     We want to preserve the original values, so we simply return them as lists.
#     """
#     hitIndexes, event_ids = zip(*batch)
#     # Optionally, convert event_ids to a tensor if you want uniform shape (they should be scalars though)
#     #event_ids_tensor = torch.tensor(event_ids, dtype=torch.long)
#     return list(hitIndexes), event_ids




class ForwardPassDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.coords, self.labels = torch.load(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        coords_tensor = self.coords[idx].clone().detach().to(dtype=torch.float)
        labels_tensor = self.labels[idx].clone().detach().to(dtype=torch.long)
        return coords_tensor, labels_tensor



class ScoringHelperDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        # Instead of unpacking into two variables, store the entire list.
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Each element of self.data is a tuple (hit_ids, event_ids) for one frame.
        hit_ids, event_ids = self.data[idx]
        # Convert them to tensors if they aren't already.
        hit_ids_tensor = torch.as_tensor(hit_ids, dtype=torch.long)
        event_ids_tensor = torch.tensor(event_ids, dtype=torch.long)
        #print("Number of labels in frame", idx, ":", len(self.labels[idx]))
        return hit_ids_tensor, event_ids_tensor
    