import os
import random
import shutil

clips = '/Users/joeljensen28/Documents/Personal/programming/DreamNet/data/clips'
train = '/Users/joeljensen28/Documents/Personal/programming/DreamNet/data/train'
test = '/Users/joeljensen28/Documents/Personal/programming/DreamNet/data/test'
val = '/Users/joeljensen28/Documents/Personal/programming/DreamNet/data/val'
sample = '/Users/joeljensen28/Documents/Personal/programming/DreamNet/data/training_sample'

def split_data(data_dir=clips, train_dir=train, test_dir=test, val_dir=val):
    data = os.listdir(data_dir)
    print('Shuffling files...')
    random.shuffle(data)
    print('Splitting data...')
    for i, file in enumerate(data):
        if i % 10 in range(0, 8): # 80% move to train
            os.rename(os.path.join(data_dir, file), os.path.join(train_dir, file))
        elif i % 10 == 8: # 10% go to test
            os.rename(os.path.join(data_dir, file), os.path.join(test_dir, file))
        elif i % 10 == 9: # 10% go to val
            os.rename(os.path.join(data_dir, file), os.path.join(val_dir, file))
        
    print('Number moved to test:')
    print(len(os.listdir(test)))
    print('Number moved to val:')
    print(len(os.listdir(val)))
    print('Number moved to train:')
    print(len(os.listdir(train)))
    print('Remaining items:')
    print(len(os.listdir(clips)))

def return_data(data_dir=clips, train_dir=train, test_dir=test, val_dir=val):
    print('Moving data...')
    for file in os.listdir(train_dir): # Move train files
        os.rename(os.path.join(train_dir, file), os.path.join(data_dir, file))
    print('Moved all from training directory')
    for file in os.listdir(test_dir): # Move train files
        os.rename(os.path.join(test_dir, file), os.path.join(data_dir, file))
    print('Moved all from test directory')
    for file in os.listdir(val_dir): # Move train files
        os.rename(os.path.join(val_dir, file), os.path.join(data_dir, file))
    print('Moved all from val directory\n')

    print('Items in data directory:')
    print(len(os.listdir(data_dir)))

def make_sample(data_dir=clips, sample_dir=sample, pct=0.001):
    n_items = len(os.listdir(data_dir))
    print(f'Files in {data_dir}: {n_items}')
    n_copying = int(pct * n_items)
    print(f'Copying {n_copying} items to {sample_dir}')
    interval = n_items // n_copying
    print(f'Interval: {interval}')

    files = os.listdir(data_dir)
    for i in range(0, n_items, interval):
        file = files[i]
        # Copy a few files into the sample dir
        shutil.copy(
            os.path.join(data_dir, file),
            os.path.join(sample_dir, file)
        )
        print(f'Copied item at index {i}')

make_sample()