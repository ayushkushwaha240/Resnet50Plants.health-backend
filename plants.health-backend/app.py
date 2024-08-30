from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from PIL import Image
from helper import load_model,pred,read_imagefile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = load_model()
classes = ["Healthy", "Powdery", "Rusted"]
# Load the model (with a check to avoid loading it multiple times)
if model is None:
    raise FileNotFoundError(f"Model file not found. Please download the model first.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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
    print(index_of_max)
    return classes[index_of_max]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
