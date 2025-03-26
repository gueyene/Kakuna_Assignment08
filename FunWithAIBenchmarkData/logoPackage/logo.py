

#logo.py

from PIL import Image
import os

def load_logo():
    """Loads and returns the Assignment08pic.jpg image resized to 200x200 pixels."""
    # Get the absolute path of the image
    image_path = os.path.join(os.path.dirname(__file__), "../LogoPicture/Assignment08pic.jpg")
    
    try:
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
