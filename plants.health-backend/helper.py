import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import requests
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup

url = os.getenv('url')

def download_file_from_google_drive(destination):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None

def load_model():
    # Get the directory of the current script
    current_directory = os.path.dirname(__file__)
    # Construct the full path for the destination file
    destination = os.path.join(current_directory, 'plantdiseasepredictionResnet50.h5')
    
    # Download the file
    download_file_from_google_drive(destination)
    
    # Load the model
    if os.path.exists(destination):
        model = tf.keras.models.load_model(destination)
        return model
    else:
        print("Model file was not found.")
        return None


    
def predict(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make a prediction
    predictions = model.predict(x)
    return predictions
