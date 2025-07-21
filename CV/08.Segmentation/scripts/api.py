# for local start: uvicorn scripts.api:app --reload

import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image
import os
from prometheus_fastapi_instrumentator import Instrumentator
from src.predict import load_model, predict_mask

# configs
MODEL_PATH = "best_model_linknet.pth"
MODEL_CONFIG_PATH = "configs/model/linknet.yaml"

# для возможности будущей доработки
# MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_linknet.pth")
# MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "configs/model/linknet.yaml")

# initialize app
app = FastAPI(
    title="Building Segmentation API",
    description="API for building semantic segmentation on satellite images.",
    version="1.0.0",
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # в проде необходимо будет указать конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create endpoint metrics
instrumentator = Instrumentator().instrument(app)

model = load_model(MODEL_PATH, MODEL_CONFIG_PATH)

# static files for our frontend
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="templates"), name="static")

# startup / shutdown
@app.on_event("startup")
async def startup_event():
    print("Starting up... Loading model.")
    try:
        app.state.model = load_model(MODEL_PATH, MODEL_CONFIG_PATH)
        print("Model load succesfully")
    except Exception as e:
        print(f"Fatal: Could not load model! Error: {e}")

    # metrics for prometheus
    instrumentator.expose(app)

# endpoints
@app.get("/", include_in_schema=False)
async def root():
    """get back main HTML page"""
    return FileResponse("templates/index.html")

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Спутниковый снимок 512х512 в формате PNG/JPEG")):
    """Take image, return binary mask"""
    # 1. read the file
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))

    # 2. predictions
    predicted_mask = await run_in_threadpool(predict_mask, model, input_image)

    # 3. convertations
    with io.BytesIO() as buf:
        predicted_mask.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    # 4. returns
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")