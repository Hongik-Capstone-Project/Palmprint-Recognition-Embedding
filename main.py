import base64
import io
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "net_params.pth"

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.ccnet import ccnet
from models.roi import PalmROIExtractor
from models.preprocessing import NormSingleROI

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model         = None
transform     = None
roi_extractor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, transform, roi_extractor

    print(f"Starting server on {DEVICE}...")
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Weight file not found: {MODEL_PATH}")

    # 모델 — 학습 시 540명, comp_weight=0.8
    model = ccnet(num_classes=540, weight=0.8)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    # 전처리 파이프라인 (ROI 추출 이후 적용)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        NormSingleROI(outchannels=1),
    ])

    # ROI 추출기 (MediaPipe + contour fallback)
    roi_extractor = PalmROIExtractor(output_size=128, grayscale=True)

    print("Model loaded successfully!")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


class PalmRequest(BaseModel):
    base64_image: str


@app.post("/embed")
def get_embedding(request: PalmRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        # base64 → PIL Image
        try:
            image_data = base64.b64decode(request.base64_image)
        except Exception:
            raise ValueError("Invalid Base64 string")

        img_pil = Image.open(io.BytesIO(image_data))

        # PIL → BGR numpy (OpenCV / MediaPipe 입력 형식)
        img_rgb = np.array(img_pil.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ROI 추출
        roi = roi_extractor.extract_with_fallback(img_bgr)
        if roi is None:
            raise ValueError("Palm ROI extraction failed: hand not detected in image")

        # ROI numpy → PIL L → transform → tensor
        pil_roi    = Image.fromarray(roi, mode="L")
        img_tensor = transform(pil_roi).unsqueeze(0).to(DEVICE)

        # 임베딩 추출 (6144-dim, L2 normalized)
        with torch.no_grad():
            embedding = model.getFeatureCode(img_tensor)

        embedding_list = embedding.squeeze().cpu().numpy().tolist()

        return {
            "status": "success",
            "embedding_dim": len(embedding_list),
            "embedding": embedding_list,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference processing failed")


@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}
