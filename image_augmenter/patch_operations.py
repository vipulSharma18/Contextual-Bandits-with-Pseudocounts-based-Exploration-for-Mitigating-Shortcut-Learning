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
        self.num_patches = (image_size[0]//patch_size)*(image_size[1]//patch_size)
        self.augmentations = transforms.Compose([
            transforms.ColorJitter(saturation=0.7,contrast=0.7), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    def query_key(self, image):
        """"Query and Key returned as a list of augmented patches """
        patches = self.extract_patches(image)
        #print(patches.size(), patches.reshape(-1, 3, self.patch_size, self.patch_size).size())
        patches = F.interpolate(patches.reshape(-1, 3, self.patch_size, self.patch_size), size=(224, 224), mode='bilinear', align_corners=False)
        #print(patches.size())
        query = self.augmentations(patches)
        key = self.augmentations(patches)
        #print("kq", query.size())
        return query, key #batch_dim, num_patches, 3, height, width

    def extract_patches(self, image):
        """ Extract non-overlapping patches from an image. """
        #batch_dim * image channels * height * width
        batch_patches = []
        image = image.unsqueeze(0) if len(image.shape)==3 else image #batch and single image handled
        unfolded = F.unfold(image, kernel_size = self.patch_size, stride=self.patch_size) #unfolding happens per color channel
        unfolded = unfolded.view(image.size(0), image.size(1), -1, self.patch_size, self.patch_size)
        return unfolded.permute(0, 2, 1, 3, 4) #batch_dim, num_patches, colors, h, w

    def mask_images(self, images, picks): 
        """Input patches and bandit's returned mask, sample from original image replacing rest with cyan masks"""
        images = images.unsqueeze(0) if len(images.shape)==3 else images
        batch_size = images.size(0)
        #directly creating a normalized image between 0 and 1
        reconstructed_images = torch.ones_like(images) * torch.tensor([0, 1, 1]).view(1, 3, 1, 1).to(images.device)
        row_patches = self.image_size[0]//self.patch_size
        col_patches = self.image_size[1]//self.patch_size
        rows = (picks//row_patches)*self.patch_size 
        cols = (picks%col_patches)*self.patch_size
        for i in range(batch_size): 
            reconstructed_images[i, :, rows[i]:rows[i]+self.patch_size, cols[i]:cols[i]+self.patch_size] = \
                images[i, :, rows[i]:rows[i]+self.patch_size, cols[i]:cols[i]+self.patch_size]
        return reconstructed_images
    
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