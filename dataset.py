import torch

class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}