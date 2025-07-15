import pandas as pd
from PIL import Image
import os 
import torch

class CUB():
    def __init__(self, root, dataset_type='train', transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        print('Load CUB datasets...')
        trn_indices, val_indices = torch.load('./solo/data_transfer/split/cub200.pth')
        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)
        df = pd.concat([df_img, df_label, df_split], axis=1)
        # relabel
        df['Label'] = df['Label'] - 1

        print('Split CUB datasets...')
        # split data
        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train' or dataset_type == 'valid':
            df = df[df['Train'] == 1]
            # split train, valid
            if dataset_type == 'train':
                df = df.iloc[trn_indices]
            else:       # dataset_type == 'valid'
                df = df.iloc[val_indices]
        else:
            raise ValueError('Unsupported dataset_type!')
        self.img_name_list = df['Image'].tolist()
        self.targets = df['Label'].tolist()
        print('Load Done...')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        image = Image.open(img_path)
        color_mode = image.mode
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target
