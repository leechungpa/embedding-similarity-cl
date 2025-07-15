"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch.utils.data as data
from PIL import Image
import torchvision.datasets as datasets
from torchvision import transforms as tf
from glob import glob

class ImageNet(datasets.ImageFolder):
    def __init__(self, root, subset = None, split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, '%s' %(split)),
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize(256)
        
        if subset is not None:
            # Read the subset of classes to include (sorted)
            with open('./datasets/split/imagenet{}.txt'.format(subset), 'r') as f:
                subdirs = f.read().splitlines()

            # Gather the files (sorted)
            imgs = []
            targets = []
            for i, subdir in enumerate(subdirs):
                subdir_path = os.path.join(self.root, subdir)
                files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
                for f in files:
                    imgs.append((f, i))
                    targets.append(i)
            self.imgs = imgs 
            self.targets = targets
            

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

