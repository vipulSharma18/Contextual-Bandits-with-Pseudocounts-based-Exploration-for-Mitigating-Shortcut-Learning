import numpy as np
from PIL import Image, ImageDraw
import random

def generate_image(circle, square, circle_size, cls):
    '''
    circle=z_s, square=z_c, circle_size=alpha_s
    '''
    # Set image size
    width, height = 224, 224
    # Create cyan background
    background_color = (0, 255, 255)  # Cyan
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Define object parameters
    min_size = 20
    alpha_2 = circle_size
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
    # Draw square
    square_color = (square, square, square)  
    square_bbox = [(x1, y1), (x1 + size_1, y1 + size_1)]
    draw.rectangle(square_bbox, fill=square_color)

    # Draw circle
    circle_color = (circle, circle, circle)
    circle_bbox = [(x2, y2), (x2 + size_2, y2 + size_2)]
    draw.ellipse(circle_bbox, fill=circle_color)

    return image

if __name__=='__main__': 
    image = generate_image(circle=120, square=120, circle_size=1, cls=0)
    image.save('greyscale.png')
    '''
    image = generate_image(circle=255, square=255, circle_size=1, cls=0)
    image.save('sample1.png')
    image = generate_image(circle=0, square=0, circle_size=1, cls=0)
    image.save('sample2.png')
    image = generate_image(circle=127, square=127, circle_size=1, cls=0)
    image.save('sample3.png')
    '''