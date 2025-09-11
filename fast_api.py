from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("last.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(input_path)

    output_path = os.path.join(OUTPUT_DIR, f"pred_{file.filename}")
    results[0].save(filename=output_path)

    return FileResponse(output_path, filename=f"pred_{file.filename}")
