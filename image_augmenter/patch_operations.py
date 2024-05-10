'''
a transform object returned
1. create patches from images and return patches
2. augment patches of the image
3. put the image back with a mask on some patches
'''
import numpy as np
import torch 
from torchvision import datasets, transforms
import random
from PIL import Image
import torch.nn.functional as F

class PatchOperations(): 
    def __init__(self, patch_size=56, image_size=(224,224)): 
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size[0]//patch_size)**2
        self.augmentations = transforms.Compose([
            transforms.ColorJitter(saturation=0.7,contrast=0.7), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    def query_key(self, image):
        """"Query and Key returned as a list of augmented patches """
        patches = self.extract_patches(image)
        queries = []
        keys = []
        for patch in patches:
            patch = F.interpolate(patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
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
    
class RemoveBackgroundTransform:
    def __init__(self, background_color=(0, 255, 255), replacement_color=(0, 0, 0)):
        self.background_color = background_color
        self.replacement_color = replacement_color

    def __call__(self, img):
        '''
        Input PIL image in torch datasets and output a PIL image.
        '''
        np_img = np.array(img)
        #print('np_img',np_img.shape) -> 224,224,3
        # Create a mask where the background color pixels are True
        mask = (np_img[:, :, 0] == self.background_color[0]) & \
               (np_img[:, :, 1] == self.background_color[1]) & \
               (np_img[:, :, 2] == self.background_color[2])
        # Replace the background color with the replacement color
        np_img[mask] = self.replacement_color
        # Convert numpy array back to PIL Image
        img = Image.fromarray(np_img)
        return img