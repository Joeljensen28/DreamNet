import torch
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokens='data/tokens', seq_len=512):
        self.tokens = tokens
        self.seq_len = seq_len
        self.index_map = []

        for filename in os.listdir(self.tokens):
            if filename.endswith('.pt'):
                file_path = os.path.join(self.tokens, filename)
                tensor = torch.load(os.path.join(self.tokens, filename))
                n_chunks = tensor.shape[0] // seq_len
                for chunk in range(n_chunks):
                    self.index_map.append((file_path, chunk*seq_len))
        
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        filepath, chunk = self.index_map[index]
        full = torch.load(filepath)
        sample = full[chunk:chunk + self.seq_len]
        input = sample[:-1]
        target = sample[1:]
        return input, target
