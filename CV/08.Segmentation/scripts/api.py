# for local start: uvicorn scripts.api:app --reload

import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import os

from src.predict import load_model, predict_mask

# configs
MODEL_PATH = "best_model_linknet.pt"

# initialize app
app = FastAPI(title="Building Segmentation API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # в проде необходимо будет указать конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_heards=["*"],
)

model = load_model(MODEL_PATH)

# static files for our frontend
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="templates"), name="statis")

# endpoints
@app.get("/", include_in_schema=False)
async def root():
    """get back main HTML page"""
    return FileResponse("templates/index.html")

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Спутниковый снимак 512х512 в формате PNG/JPEG")):
    """Take image, return binary mask"""
    # 1. read the file
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))

    # 2. predictions
    predicted_mask = predict_mask(model, input_image)

    # 3. convertations
    with io.BytesIO() as buf:
        predicted_mask.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    # 4. returns
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")