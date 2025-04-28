from modules.Dataset.Dataset import Dataset
from torch.utils.data import DataLoader

data = Dataset()
loader = DataLoader(
    data,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

for epoch in range(1):
    for batch_inputs, batch_targets in loader:
        print(batch_inputs[2][0])
        break