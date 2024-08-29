from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from io import BytesIO
import os
from helper import load_model

app = FastAPI()
model = load_model()

# Load the model (with a check to avoid loading it multiple times)
if model is None:
    raise FileNotFoundError(f"Model file not found. Please download the model first.")

# Define class labels
class_labels = ['Healthy', 'Powdery', 'Rust']

@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        with open("./Space.talk/index.html", "r") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>HTML file not found</h1>", status_code=404)

@app.post("/upload", response_class=JSONResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
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
        
        # Return the result as a JSON response
        return JSONResponse(content={"predicted_label": predicted_label}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
