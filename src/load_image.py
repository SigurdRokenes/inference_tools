import numpy as np
from io import BytesIO
from tensorflow import io
from PIL import Image

def load_image_to_numpy_array(path):
    """
    Load a single image to numpy array
    """
    img_data = io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(im_height, im_width, 3).astype(np.uint8)