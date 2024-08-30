from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from fastapi.responses import StreamingResponse
from helper import load_model,pred,read_imagefile
import uuid

app = FastAPI()
model = load_model()
classes = ["Healthy", "Powdery", "Rust"]
# Load the model (with a check to avoid loading it multiple times)
if model is None:
    raise FileNotFoundError(f"Model file not found. Please download the model first.")

# @app.get("/", response_class=HTMLResponse)
# async def get_html():
#     try:
#         with open("./Space.talk/index.html", "r") as file:
#             return file.read()
#     except FileNotFoundError:
#         return HTMLResponse(content="<h1>HTML file not found</h1>", status_code=404)

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = pred(model,image)
    print(prediction)
    output = np.array(prediction)
    # Get the index of the maximum value
    index_of_max = np.argmax(output)
    return classes[index_of_max]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
