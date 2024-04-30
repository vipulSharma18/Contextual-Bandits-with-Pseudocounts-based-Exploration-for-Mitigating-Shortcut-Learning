import numpy as np
from PIL import Image, ImageDraw
import random

def generate_image():
    # Set image size
    width, height = 224, 224
    # Create cyan background
    background_color = (0, 255, 255)  # Cyan
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Define object parameters
    min_size = 20
    sizes = [1, 2, 3, 4, 5]
    alpha_2 = random.choice(sizes)
    size_1 = min_size
    size_2 = min_size * alpha_2

    # Ensure non-overlapping
    x1 = random.randint(0, width - size_1)
    y1 = random.randint(0, height - size_1)
    x2 = random.randint(0, width - size_2)
    y2 = random.randint(0, height - size_2)
    for i in range(10): 
        # Check for overlap and adjust positions if necessary
        if (x1 < x2 + size_2 and x1 + size_1 > x2 and
            y1 < y2 + size_2 and y1 + size_1 > y2):
            # Squares overlap, adjust position of square 2
            x2 = random.randint(0, width - size_2)
            y2 = random.randint(0, height - size_2)
        elif (x2 < x1 + size_1 and x2 + size_2 > x1 and
            y2 < y1 + size_1 and y2 + size_2 > y1):
            # Squares overlap, adjust position of square 1
            x1 = random.randint(0, width - size_1)
            y1 = random.randint(0, height - size_1)
        else: 
            break
    
    # Generate random color vectors
    # get the logic of how to convert z_s and z_c into a grayscale value of color from Thomas
    z_1 = 160 #np.random.normal(0, 1, 100)
    z_2 = 100 #np.random.randint(0, 1, 100)

    # Draw square
    square_color = (z_1, z_1, z_1)  # Same grayscale value for entire square
    square_bbox = [(x1, y1), (x1 + size_1, y1 + size_1)]
    draw.rectangle(square_bbox, fill=square_color)

    # Draw circle
    circle_color = (z_2, z_2, z_2)  # Same grayscale value for entire circle
    circle_bbox = [(x2, y2), (x2 + size_2, y2 + size_2)]
    draw.ellipse(circle_bbox, fill=circle_color)

    return image

# Generate and display the image
image = generate_image()
image.show()
