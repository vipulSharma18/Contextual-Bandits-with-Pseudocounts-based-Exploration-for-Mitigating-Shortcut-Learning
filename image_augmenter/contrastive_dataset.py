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

class PatchOperations(): 
    def __init__(self, patch_size=56, image_size=224): 
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size//patch_size)**2
        self.augmentations = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=45),
            ])
        ])
    def query_key(self, image):
        """"Query and Key returned as a list of augmented patches """
        patches = self.extract_patches(image)
        # Apply two different augmentations
        results = []
        for patch in patches: 
            query = self.augmentations(patch)
            key = self.augmentations(patch)
            results.append((query, key))
        return results
    def extract_patches(self, image):
        """ Extract non-overlapping patches from an image. """
        patches = [image.crop((i % 4 * self.patch_size, i // 4 * self.patch_size,
                               (i % 4 + 1) * self.patch_size, (i // 4 + 1) * self.patch_size))
                   for i in range(self.num_patches)]
        return patches
    def reconstruct_image(self, patches, mask): 
        """Input patches and bandit's returned mask, recreate original image with cyan masks"""
        image = Image.new('RGB', self.image_size)
        idx = 0
        for i in range(0, self.image_size[0], self.patch_size):
            for j in range(0, self.image_size[1], self.patch_size):
                patch = patches[idx]
                if idx in mask:
                    # Create a cyan patch for masking
                    blue_patch = Image.new('RGB', (self.patch_size, self.patch_size), (0, 255, 255))
                    image.paste(blue_patch, (i, j))
                else:
                    image.paste(patch, (i, j))
                idx += 1
        return image
    
