import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from io import BytesIO
from PIL import Image
# from down import download

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def load_model():
    model = tf.keras.models.load_model('plantdesiesepredictionResnet50.h5')
    print("model loaded")
    return model
    
def pred(model,image: Image.Image):
    if model is None:
        model = load_model()
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0
    return model.predict(image)
