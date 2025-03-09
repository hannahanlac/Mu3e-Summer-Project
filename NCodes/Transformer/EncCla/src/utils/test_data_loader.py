import torch
from torch.utils.data import DataLoader
import os
import toml
import data_utils  # Make sure this module is in your PYTHONPATH or same folder

def test_dataloader():
    # Define a minimal config dictionary for testing.
    # Update the 'data_dir' and file names as appropriate for your setup.
    config = {
        "data": {
            "data_dir": "/root/Mu3eProject/RawData/TransformerData/TrainingData",  # Replace with the directory containing your .pt files
            "test_file": "signal1_96_sorted_test.pt",
            "test_helperfile": "signal1_96_sorted_helper.pt",
            "dataloader_num_workers": 0  # Set to 0 for debugging purposes
        },
        "training": {
            "batch_size": 2,   # Use a small batch size for testing
            "shuffle": False
        }
    }
    
    device = "cpu"  # For testing/running locally
    
    # Load the dataloaders in evaluation mode
    loaders = data_utils.load_dataloader(config, device, mode="eval")
    test_loader = loaders["test"]
    test_helper_loader = loaders["test_helper"]
    
    print("Testing the test dataloader:")
    for batch in test_loader:
        # According to your collate_fn, each batch is (features_padded, labels_padded)
        features, labels = batch
        print("Features shape:", features.shape)  # e.g., (batch_size, max_hits_in_batch, 3)
        print("Labels shape:", labels.shape)      # e.g., (batch_size, max_hits_in_batch)
        break  # Only test one batch
    
    print("\nTesting the test helper dataloader:")
    for batch in test_helper_loader:
        hit_ids, event_ids = batch
        print("Hit IDs shape:", hit_ids.shape)
        print("Event IDs shape:", event_ids.shape)
        break

if __name__ == "__main__":
    test_dataloader()
