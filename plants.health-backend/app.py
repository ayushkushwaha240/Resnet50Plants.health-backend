from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from io import BytesIO
from helper import load_model

app = FastAPI()

# Load the model
model = tf.keras.models.load_model('plantdesiesepredictionResnet50.h5')

# Define class labels
class_labels = ['Healthy', 'Powdery', 'Rust']

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("./Space.talk/index.html", "r") as file:
        return file.read()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Read image file
    img_bytes = await file.read()
    img = image.load_img(BytesIO(img_bytes), target_size=(224, 224))
    
    # Preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make a prediction
    predictions = model.predict(x)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]
    
    # Return the result
    return predicted_label
