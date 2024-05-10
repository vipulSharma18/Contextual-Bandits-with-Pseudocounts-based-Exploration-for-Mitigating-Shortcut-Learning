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
    def __init__(self, patch_size=56, image_size=(224,224)): 
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size[0]//patch_size)**2
        self.augmentations = transforms.Compose([
                transforms.RandomVerticalFlip(), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20)
        ])
    def query_key(self, image):
        """"Query and Key returned as a list of augmented patches """
        patches = self.extract_patches(image)
        queries = []
        keys = []
        for patch in patches:
            query = self.augmentations(patch)
            key = self.augmentations(patch)
            queries.append(query)
            keys.append(key)
        return torch.stack(queries), torch.stack(keys)
    def extract_patches(self, image):
        """ Extract non-overlapping patches from an image. """
        #image channels * height * width
        patches = []
        for i in range(0, image.size(1), self.patch_size):  # Iterate over height
            for j in range(0, image.size(2), self.patch_size):  # Iterate over width
                patch = image[:, i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)
        return patches
    def reconstruct_image(self, patches, mask): 
        """Input patches and bandit's returned mask, recreate original image with cyan masks"""
        image = Image.new('RGB', self.image_size)
        idx = 0
        for i in range(0, self.image_size[0], self.patch_size):
            for j in range(0, self.image_size[1], self.patch_size):
                
                if idx in mask:
                    # Create a cyan patch for masking
                    patch = Image.new('RGB', (self.patch_size, self.patch_size), (0, 255, 255))
                else:
                    patch = transforms.ToPIL()(patches[idx])
                image.paste(patch, (i, j))
                idx += 1
        return transforms.ToTensor()(image)
    
