'''
a transform object returned
1. create patches from images and return patches
2. augment patches of the image
3. put the image back with a mask on some patches
'''
import torch 
from torchvision import datasets, transforms
import random
from PIL import Image

class ContrastivePatchDataset(torch.utils.data.Dataset): 
    def __init__(self, root_dir, transform=None, patch_size=56): 
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.patch_size = patch_size
        self.augmentations = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=45),
            ])
        ])
    def __len__(self):
        return len(self.dataset) * (224 // self.patch_size) ** 2
    def __getitem__(self, idx):
        img_idx = idx // ((224 // self.patch_size) ** 2)
        patch_idx = idx % ((224 // self.patch_size) ** 2)
        image, label = self.dataset[img_idx]
        patches = self.extract_patches(image)
        patch = patches[patch_idx]
        # Apply two different augmentations
        query = self.augmentations(patch)
        key = self.augmentations(patch)
        return query, key, label
    def extract_patches(self, image):
        """ Extract non-overlapping patches from an image. """
        patches = []
        for i in range(0, image.size[0], self.patch_size):
            for j in range(0, image.size[1], self.patch_size):
                patch = image.crop((i, j, i + self.patch_size, j + self.patch_size))
                patches.append(patch)
        return patches

