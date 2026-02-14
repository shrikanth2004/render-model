import os

from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import csv
from io import BytesIO
from PIL import Image
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
MODEL_FILE = os.environ.get(
    "MODEL_FILE",
    str(ROOT_DIR / "plant_disease_model_augmented.h5")
)
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 224))
CLASS_NAMES_FILE = os.environ.get("CLASS_NAMES_FILE", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api")

DISEASE_INFO = {}
try:
    csv_path = ROOT_DIR / "disease_info.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                disease_name = row.get("disease_name")
                if disease_name:
                    DISEASE_INFO[disease_name] = {
                        "description": row.get("description"),
                        "possible_steps": row.get("possible_steps"),
                        "image_url": row.get("image_url"),
                    }
except Exception as e:
    logger.error(f"Error loading disease_info.csv: {str(e)}")


class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    description: Optional[str] = None
    possible_steps: Optional[str] = None
    image_url: Optional[str] = None


STATUS_STORE: List[dict] = []
PREDICTIONS_STORE: List[dict] = []

MODEL = None
MODEL_LOAD_ERROR = None
CLASS_NAMES: List[str] = []


def load_class_names():
    global CLASS_NAMES

    if CLASS_NAMES_FILE:
        p = Path(CLASS_NAMES_FILE)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                CLASS_NAMES = [
                    line.strip() for line in f.readlines()
                    if line.strip()
                ]
            return

    if DISEASE_INFO:
        CLASS_NAMES = list(DISEASE_INFO.keys())


def load_keras_model():
    global MODEL, MODEL_LOAD_ERROR

    if load_model is None:
        MODEL_LOAD_ERROR = "TensorFlow/Keras not installed"
        logger.error(MODEL_LOAD_ERROR)
        return

    model_path = Path(MODEL_FILE)
    logger.info(f"Loading model from: {model_path.absolute()}")

    if not model_path.exists():
        MODEL_LOAD_ERROR = f"Model file not found: {model_path.absolute()}"
        logger.error(MODEL_LOAD_ERROR)
        return

    try:
        MODEL = load_model(
            str(model_path),
            compile=False
        )
        MODEL_LOAD_ERROR = None
        logger.info("Model loaded successfully")

    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        MODEL = None
        logger.error(f"Model load failed: {e}", exc_info=True)


def preprocess_image(contents: bytes) -> np.ndarray:
    img = Image.open(BytesIO(contents))

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img).astype("float32")

    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)

    return arr


@api_router.get("/")
async def root():
    return {"message": "Backend running"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    obj = StatusCheck(**input.model_dump())
    STATUS_STORE.append(obj.model_dump())
    return obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    return STATUS_STORE


@api_router.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be image")

    contents = await file.read()

    if MODEL is None:
        raise HTTPException(
            500,
            f"Model not loaded. Reason: {MODEL_LOAD_ERROR}"
        )

    input_arr = preprocess_image(contents)

    preds = MODEL.predict(input_arr)

    preds = preds[0]

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if CLASS_NAMES and class_idx < len(CLASS_NAMES):
        predicted_class = CLASS_NAMES[class_idx]
    else:
        predicted_class = str(class_idx)

    info = DISEASE_INFO.get(predicted_class, {})

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        description=info.get("description"),
        possible_steps=info.get("possible_steps"),
        image_url=info.get("image_url"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_class_names()
    load_keras_model()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
