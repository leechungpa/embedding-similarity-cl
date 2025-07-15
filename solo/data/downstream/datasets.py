import os
import json

import torch
from PIL import Image

class ImageList(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)
    

class Pets(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'annotations', f'{split}.txt')) as f:
            annotations = [line.split() for line in f]

        samples = []
        for sample in annotations:
            path = os.path.join(root, 'images', sample[0] + '.jpg')
            label = int(sample[1])-1
            samples.append((path, label))

        super().__init__(samples, transform)
        
class Food101(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'meta', 'classes.txt')) as f:
            classes = [line.strip() for line in f]
        with open(os.path.join(root, 'meta', f'{split}.json')) as f:
            annotations = json.load(f)

        samples = []
        for i, cls in enumerate(classes):
            for path in annotations[cls]:
                samples.append((os.path.join(root, 'images', f'{path}.jpg'), i))

        super().__init__(samples, transform)

class DTD(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'labels', f'{split}1.txt')) as f:
            paths = [line.strip() for line in f]

        classes = sorted(os.listdir(os.path.join(root, 'images')))
        samples = [(os.path.join(root, 'images', path), classes.index(path.split('/')[0])) for path in paths]
        super().__init__(samples, transform)
        

class SUN397(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, 'ClassName.txt')) as f:
            classes = [line.strip() for line in f]

        with open(os.path.join(root, f'{split}_01.txt')) as f:
            samples = []
            for line in f:
                path = line.strip()
                for y, cls in enumerate(classes):
                    if path.startswith(cls+'/'):
                        samples.append((os.path.join(root, path[1:]), y))
                        break
        super().__init__(samples, transform)
        
class MIT67(ImageList):
    def __init__(self, root, split, transform=None):
        with open(os.path.join(root, f'{split}Images.txt')) as f:
            paths = [line.strip() for line in f]
        classes = sorted(os.listdir(os.path.join(root,'Images')))
        samples = [(os.path.join(root, 'Images', path), classes.index(path.split('/')[0])) for path in paths]
        super().__init__(samples, transform)