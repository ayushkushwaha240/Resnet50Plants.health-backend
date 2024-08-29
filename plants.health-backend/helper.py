import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import requests

def load_model():
    model = tf.keras.models.load_model('plantdesiesepredictionResnet50.h5')
    return model
    
def predict(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make a prediction
    predictions = model.predict(x)
    return predictions
