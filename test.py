from modules.Dataset.Dataset import Dataset
from modules.Model.model import AudioTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
from torch.optim.lr_scheduler import StepLR
import os
import random
from torch.cuda import is_available, manual_seed_all
import numpy as np
import torch

torch.manual_seed(777)
np.random.seed(777)

device = 'cpu'
num_workers = 0
if is_available():
    device = 'cuda'
    manual_seed_all(777)
    torch.backends.cudnn.benchmark = True
    num_workers = max(1, os.cpu_count() - 1)

print(f'Using {device}')

TOKENS_DIR = '/content/drive/MyDrive/dreamnet/data/tokens'
all_files = sorted(f for f in os.listdir(TOKENS_DIR) if f.endswith('.pt'))
random.seed(777)
random.shuffle(all_files)
split_idx = int(0.1 * len(all_files))
val_files = all_files[:split_idx]
train_files = all_files[split_idx:]

train_data = Dataset(tokens=TOKENS_DIR)
val_data = Dataset(tokens=TOKENS_DIR)

train_data.index_map = [
    (fp, off) for (fp, off) in train_data.index_map
    if os.path.basename(fp) in train_files
]

val_data.index_map = [
    (fp, off) for (fp, off) in val_data.index_map
    if os.path.basename(fp) in val_files
]

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False
)

model = AudioTransformer(
    seq_len=511
)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(1, 51):
    print(f'Number of batches: {len(train_loader)}')
    print(f'Epoch {epoch}')
    model.train()
    for batch_inputs, batch_targets in train_loader:
        batch_start = time.time()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        logits = model(batch_inputs)

        B, L, K, V = logits.shape

        total_loss = 0.0
        for i in range(K):
            logits_i = logits[:, :, i, :].reshape(-1, V)
            targets_i = batch_targets[:, :, i].reshape(-1)
            total_loss += loss_fn(logits_i, targets_i)

        loss = total_loss / K

        print(f'Batch loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f'Batch processing time: {(time.time() - batch_start):.2f}s')
    scheduler.step()

    print(f'Epoch {epoch} loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch_inputs, batch_targets in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_inputs)

            B, L, K, V = logits.shape

            total_loss = 0.0
            for i in range(K):
                logits_i = logits[:, :, i, :].reshape(-1, V)
                targets_i = batch_targets[:, :, i].reshape(-1)
                total_loss += loss_fn(logits_i, targets_i)

            loss = total_loss / K
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Val loss: {val_loss}')
    
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'sched_state': scheduler.state_dict()
    }, f'ckpt_epoch{epoch:02d}.pt')